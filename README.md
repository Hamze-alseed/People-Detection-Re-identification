# 🕵️ People Tracking System

Real-time **Person Detection, Tracking & Re-Identification System** using:
- YOLOv8
- DeepSORT
- ResNet50
- Streamlit
- OpenCV
- PyTorch

Built for real-time webcam tracking with GPU acceleration support.

---

## 🚀 Features
- Real-time person detection
- Multi-person tracking
- Person re-identification
- Feature-based matching
- GPU acceleration (CUDA)
- Live dashboard
- FPS monitoring
- Historical tracking data
- Streamlit web interface

---

## 🧠 AI Architecture

```text
Camera → YOLOv8 → DeepSORT → ResNet50 Feature Extractor
                      ↓
              Re-ID Matching System
                      ↓
               Streamlit Dashboard


| Component | Tech      |
| --------- | --------- |
| Detection | YOLOv8    |
| Tracking  | DeepSORT  |
| Features  | ResNet50  |
| Backend   | Python    |
| UI        | Streamlit |
| CV        | OpenCV    |
| ML        | PyTorch   |


⚙️ Installation

git clone https://github.com/YOUR_USERNAME/people-tracking-system.git
cd people-tracking-system
pip install -r requirements.txt

▶️ Run

streamlit run app.py
