import streamlit as st
import cv2
import numpy as np
import time
from datetime import datetime
import torch
import torchvision.transforms as T
from torchvision.models import resnet50
from ultralytics import YOLO
from collections import defaultdict, deque
from sklearn.metrics.pairwise import cosine_similarity
import sys
import subprocess
import importlib
import warnings
warnings.filterwarnings('ignore')

# -----------------------------------------
# 1️⃣ Main inteface settings
# -----------------------------------------
st.set_page_config(page_title="People Tracking System", layout="wide")

st.title("🕵️‍♂️ People Tracking System")
st.write("Real-time Person Detection & Re-identification")

# -----------------------------------------
# 2️⃣ Installing libraries
# -----------------------------------------
def check_and_install_packages():
    """Check and install required packages"""
    required_packages = [
        'deep-sort-realtime',
        'ultralytics',
        'torch',
        'torchvision',
        'opencv-python',
        'scikit-learn',
        'streamlit'
    ]
    
    for package in required_packages:
        try:
            if package == 'deep-sort-realtime':
                import deep_sort_realtime
            elif package == 'ultralytics':
                import ultralytics
            elif package == 'torch':
                import torch
            elif package == 'torchvision':
                import torchvision
            elif package == 'opencv-python':
                import cv2
            elif package == 'scikit-learn':
                import sklearn
            elif package == 'streamlit':
                import streamlit
        except ImportError:
            with st.spinner(f"Installing {package}..."):
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Try to install missing packages
try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
except ImportError:
    with st.warning("Installing required packages... This may take a moment."):
        check_and_install_packages()
        from deep_sort_realtime.deepsort_tracker import DeepSort

# -----------------------------------------
# 3️⃣ (State)
# -----------------------------------------
if "running" not in st.session_state:
    st.session_state.running = False
if "people_data" not in st.session_state:
    st.session_state.people_data = {}
if "total_unique_people" not in st.session_state:
    st.session_state.total_unique_people = 0
if "known_persons" not in st.session_state:
    st.session_state.known_persons = []
if "model_initialized" not in st.session_state:
    st.session_state.model_initialized = False
if "frame_placeholder" not in st.session_state:
    st.session_state.frame_placeholder = None
if "metrics_placeholder" not in st.session_state:
    st.session_state.metrics_placeholder = None
if "data_placeholder" not in st.session_state:
    st.session_state.data_placeholder = None
if "device_type" not in st.session_state:
    st.session_state.device_type = None

# -----------------------------------------
# 4️⃣ Initialize models
# -----------------------------------------
@st.cache_resource
def initialize_models():
    """Initialize all models once and cache them"""
    # Initialize device - FORCE GPU if available
    if torch.cuda.is_available():
        device = torch.device("cuda")
        st.success("✅ GPU Detected! Using CUDA acceleration")
    else:
        device = torch.device("cpu")
        st.warning("⚠️ GPU not available. Using CPU (slower)")
    
    st.session_state.device_type = device.type
    
    # Initialize Feature Extractor
    class FeatureExtractor(torch.nn.Module):
        def __init__(self):
            super(FeatureExtractor, self).__init__()
            resnet = resnet50(weights='DEFAULT')
            self.features = torch.nn.Sequential(*list(resnet.children())[:-2])
            self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
            self.fc = torch.nn.Sequential(
                torch.nn.Linear(2048, 512),
                torch.nn.BatchNorm1d(512),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.5),
                torch.nn.Linear(512, 256)
            )
            
        def forward(self, x):
            x = self.features(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
            return torch.nn.functional.normalize(x, p=2, dim=1)
    
    resnet = FeatureExtractor().to(device).eval()
    
    # Transformation
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((256, 128)),
        T.RandomHorizontalFlip(p=0.5),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Initialize YOLO with explicit device
    try:
        model = YOLO("yolov8n.pt").to(device)
        # Test if model is on GPU
        if device.type == 'cuda':
            # Move a dummy tensor to GPU to test
            test_tensor = torch.randn(1, 3, 640, 640).to(device)
            st.success(f"✅ YOLO model successfully loaded on {torch.cuda.get_device_name(0)}")
    except Exception as e:
        st.error(f"Error loading YOLO on GPU: {e}")
        # Fallback to CPU
        device = torch.device("cpu")
        st.session_state.device_type = "cpu"
        model = YOLO("yolov8n.pt").to(device)
    
    tracker = DeepSort(max_age=20, nn_budget=100, max_cosine_distance=0.4, max_iou_distance=0.7)
    
    return device, resnet, transform, model, tracker

# -----------------------------------------
# 5️⃣ Helper Functions
# -----------------------------------------
def extract_deep_features(frame, bbox, resnet, transform, device):
    """Extract deep features using ResNet"""
    x1, y1, x2, y2 = map(int, bbox)
    x1, y1 = max(0, x1-10), max(0, y1-10)
    x2, y2 = min(frame.shape[1], x2+10), min(frame.shape[0], y2+10)
    
    person_roi = frame[y1:y2, x1:x2]
    if person_roi.size == 0:
        return None
    
    try:
        img = transform(person_roi).unsqueeze(0).to(device)
        with torch.no_grad():
            features = resnet(img).squeeze().cpu().numpy()
        return features / (np.linalg.norm(features) + 1e-12)
    except Exception as e:
        return None

def extract_clothing_histogram(frame, bbox):
    """Extract clothing color histogram"""
    x1, y1, x2, y2 = map(int, bbox)
    height = y2 - y1
    lower_y1 = y1 + int(0.4 * height)
    torso = frame[lower_y1:y2, x1:x2]
    if torso.size == 0:
        return None
    
    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
    histograms = []
    h, w = torso.shape[:2]
    
    for i in range(2):
        for j in range(2):
            y_start = i * h // 2
            y_end = (i + 1) * h // 2
            x_start = j * w // 2
            x_end = (j + 1) * w // 2
            cell = hsv[y_start:y_end, x_start:x_end]
            hist = cv2.calcHist([cell], [0, 1], None, [8, 8], [0, 180, 0, 256])
            histograms.append(cv2.normalize(hist, hist).flatten())
    
    return np.concatenate(histograms)

def compare_features(feat1, feat2):
    """Compare two feature vectors"""
    if feat1 is None or feat2 is None:
        return 0
    
    deep_sim = cosine_similarity([feat1[0]], [feat2[0]])[0][0] if feat1[0] is not None and feat2[0] is not None else 0
    hist_sim = cv2.compareHist(feat1[1].astype(np.float32), feat2[1].astype(np.float32), cv2.HISTCMP_CORREL) if feat1[1] is not None and feat2[1] is not None else 0
    return 0.7 * deep_sim + 0.3 * hist_sim

# -----------------------------------------
# 6️⃣ Main tracking function
# -----------------------------------------
def run_tracking():
    """Main tracking function"""
    try:
        # Initialize models if not already done
        if not st.session_state.model_initialized:
            with st.spinner("Loading models..."):
                device, resnet, transform, model, tracker = initialize_models()
                st.session_state.model_initialized = True
                st.session_state.models = {
                    'device': device,
                    'resnet': resnet,
                    'transform': transform,
                    'model': model,
                    'tracker': tracker
                }
                # Display GPU info
                if device.type == 'cuda':
                    gpu_name = torch.cuda.get_device_name(0)
                    st.sidebar.success(f"GPU: {gpu_name}")
                else:
                    st.sidebar.warning("Running on CPU")
        else:
            models = st.session_state.models
            device = models['device']
            resnet = models['resnet']
            transform = models['transform']
            model = models['model']
            tracker = models['tracker']
        
        # Initialize tracking variables
        pending_features = defaultdict(lambda: {'deep': deque(maxlen=25), 'clothing': deque(maxlen=25)})
        
        # Open webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("❌ Unable to access webcam!")
            return
        
        frame_count = 0
        start_time = time.time()
        fps_values = []
        
        # Create a container for the video frame
        frame_container = st.empty()
        
        # Main tracking loop
        while st.session_state.running:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to read from webcam!")
                break
            
            frame_count += 1
            
            # Run YOLO detection with explicit device handling
            try:
                # Use half precision on GPU for better performance
                if device.type == 'cuda':
                    results = model(frame, classes=[0], conf=0.7, verbose=False, half=True)
                else:
                    results = model(frame, classes=[0], conf=0.7, verbose=False)
            except Exception as e:
                st.error(f"Detection error: {e}")
                continue
            
            detections = []
            
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                w, h = x2 - x1, y2 - y1
                if h < 50 or h > 500 or w/h < 0.3 or w/h > 1.5:
                    continue
                detections.append([[x1, y1, w, h], box.conf.item()])
            
            # Update tracker
            tracks = tracker.update_tracks(detections, frame=frame) if detections else []
            current_persons = set()
            
            for track in tracks:
                if not track.is_confirmed():
                    continue
                
                x1, y1, x2, y2 = map(int, track.to_ltrb())
                deepsort_id = track.track_id
                
                # Extract features
                deep_feat = extract_deep_features(frame, (x1, y1, x2, y2), resnet, transform, device)
                clothing_feat = extract_clothing_histogram(frame, (x1, y1, x2, y2))
                
                # Update pending features
                if deep_feat is not None:
                    pending_features[deepsort_id]['deep'].append(deep_feat)
                if clothing_feat is not None:
                    pending_features[deepsort_id]['clothing'].append(clothing_feat)
                
                # Wait for stable features
                min_samples = 5
                if len(pending_features[deepsort_id]['deep']) < min_samples or len(pending_features[deepsort_id]['clothing']) < min_samples:
                    continue
                
                # Get current features
                deep_features = list(pending_features[deepsort_id]['deep'])
                clothing_features = list(pending_features[deepsort_id]['clothing'])
                
                current_features = (
                    np.mean(deep_features, axis=0) if deep_features else None,
                    np.mean(clothing_features, axis=0) if clothing_features else None
                )
                
                # Person matching
                best_match_id = None
                best_match_score = 0.75  # Threshold
                
                for known in st.session_state.known_persons:
                    score = compare_features(current_features, (known["features"], known["clothing"]))
                    if score > best_match_score:
                        best_match_score = score
                        best_match_id = known["id"]
                
                if best_match_id is not None:
                    person_id = best_match_id
                    # Update existing person
                    for known in st.session_state.known_persons:
                        if known["id"] == person_id:
                            if current_features[0] is not None and known["features"] is not None:
                                known["features"] = 0.1 * current_features[0] + 0.9 * known["features"]
                            if current_features[1] is not None and known["clothing"] is not None:
                                known["clothing"] = 0.1 * current_features[1] + 0.9 * known["clothing"]
                            known["last_seen"] = datetime.now().strftime("%H:%M:%S")
                            known["count"] += 1
                            break
                else:
                    # New person
                    st.session_state.total_unique_people += 1
                    person_id = st.session_state.total_unique_people
                    new_person = {
                        "id": person_id,
                        "features": current_features[0],
                        "clothing": current_features[1],
                        "first_seen": datetime.now().strftime("%H:%M:%S"),
                        "last_seen": datetime.now().strftime("%H:%M:%S"),
                        "count": 1
                    }
                    st.session_state.known_persons.append(new_person)
                    st.session_state.people_data[person_id] = {
                        "First Seen": new_person["first_seen"],
                        "Last Seen": new_person["last_seen"],
                        "Detection Count": new_person["count"],
                        "Status": "Active"
                    }
                
                current_persons.add(person_id)
                
                # Draw bounding box and ID
                color = (0, 255, 0)  # Green
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"ID: {person_id}", (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Calculate FPS
            current_time = time.time()
            fps = 1.0 / (current_time - start_time) if frame_count > 1 else 0
            fps_values.append(fps)
            start_time = current_time
            
            # Keep only last 10 fps values
            if len(fps_values) > 10:
                fps_values.pop(0)
            
            avg_fps = np.mean(fps_values) if fps_values else 0
            
            # Display metrics on frame
            cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, f"Current: {len(current_persons)}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, f"Total: {st.session_state.total_unique_people}", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, f"Device: {st.session_state.device_type.upper()}", (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Convert to RGB for Streamlit
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Update the frame - FIXED the deprecated parameter
            frame_container.image(frame_rgb, channels="RGB", use_container_width=True)
            
            # Update metrics
            update_display_metrics(avg_fps, len(current_persons))
            
            # Update data table
            update_data_table()
            
            # Small delay to prevent freezing
            time.sleep(0.01)
        
        # Release resources
        cap.release()
        cv2.destroyAllWindows()
        
    except Exception as e:
        st.error(f"Error in tracking: {str(e)}")
        import traceback
        st.error(traceback.format_exc())

def update_display_metrics(fps, current_persons):
    """Update the metrics display"""
    if st.session_state.metrics_placeholder:
        device_text = "GPU" if st.session_state.device_type == 'cuda' else "CPU"
        metrics_text = f"""
        ### 📊 Live Metrics
        - **FPS:** {fps:.1f}
        - **Active Persons:** {current_persons}
        - **Total Unique Persons:** {st.session_state.total_unique_people}
        - **Device:** {device_text}
        """
        st.session_state.metrics_placeholder.markdown(metrics_text)

def update_data_table():
    """Update the people data table"""
    if st.session_state.people_data and st.session_state.data_placeholder:
        import pandas as pd
        df = pd.DataFrame.from_dict(st.session_state.people_data, orient='index')
        st.session_state.data_placeholder.dataframe(df, use_container_width=True)

# -----------------------------------------
# 7️⃣ Main interface
# -----------------------------------------

# Create layout
st.subheader("🎥 Live Camera Feed")

# Control buttons section
st.subheader("🎛️ Controls")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("▶ Start Tracking", type="primary", use_container_width=True, 
                key="start_button"):
        st.session_state.running = True
        st.success("🚀 Tracking started!")
        st.rerun()

with col2:
    if st.button("⏹ Stop", type="secondary", use_container_width=True, 
                key="stop_button"):
        st.session_state.running = False
        st.warning("⛔ Tracking stopped.")
        st.rerun()

with col3:
    if st.button("🗑️ Clear Data", type="secondary", use_container_width=True,
                key="clear_data_button"):
        st.session_state.people_data = {}
        st.session_state.total_unique_people = 0
        st.session_state.known_persons = []
        st.success("Data cleared!")
        st.rerun()

# Main content area
col_left, col_right = st.columns([2, 1])

with col_left:
    # Video feed will be displayed here when tracking starts
    if st.session_state.running:
        # Initialize tracking
        run_tracking()
    else:
        # Display placeholder when not running
        st.info("Click 'Start Tracking' to begin live tracking")
        # You can also show a static image or instructions here

with col_right:
    st.subheader("📊 Statistics")
    # Create metrics placeholder
    st.session_state.metrics_placeholder = st.empty()
    
    st.subheader("👥 Tracked People")
    st.session_state.data_placeholder = st.empty()
    if not st.session_state.people_data:
        st.info("No people tracked yet. Start tracking to see data here.")

# -----------------------------------------
# 8️⃣ System Info
# -----------------------------------------
st.sidebar.title("ℹ️ System Information")

# Check GPU status
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    st.sidebar.success(f"✅ GPU Available: {gpu_name}")
    st.sidebar.write(f"CUDA Version: {torch.version.cuda}")
else:
    st.sidebar.warning("⚠️ No GPU detected. Using CPU.")

st.sidebar.markdown("""
### About This System
This real-time people tracking system uses:
- **YOLOv8** for person detection
- **DeepSORT** for tracking
- **ResNet50** for feature extraction
- **Streamlit** for visualization

### Features:
- Real-time person detection
- Person re-identification
- Feature-based matching
- Live metrics display
- Historical tracking data
""")

# Performance tips
st.sidebar.markdown("---")
st.sidebar.markdown("### ⚡ Performance Tips")
st.sidebar.markdown("""
1. Ensure good lighting
2. Stand clearly in camera view
3. Close other GPU-intensive apps
4. Use wired internet connection
""")

# System requirements
st.sidebar.markdown("---")
st.sidebar.markdown("### 💻 System Requirements")
st.sidebar.markdown("""
- **Minimum:** 8GB RAM, CPU
- **Recommended:** 16GB RAM, NVIDIA GPU
- **Webcam:** 720p or higher
- **OS:** Windows 10/11, Linux, macOS
""")

# Add a footer
st.sidebar.markdown("---")
st.sidebar.caption("Phoenix Tracking System v1.0")