# 🕵️‍♂️ People Detection & Re-Identification System

> Real-time multi-person detection, tracking, and re-identification system built for real-world AI applications using deep learning, computer vision, and real-time inference pipelines.

---

## 🚀 Overview

This project is a **real-time AI system** that performs:
-  Human detection  
-  Multi-object tracking  
-  Person re-identification  
-  Feature-based identity matching  
-  Real-time visualization  
-  GPU-accelerated inference  

Built as a **full AI pipeline**, not just a model demo.

---

## 🎯 Key Features

-  Real-time person detection (YOLOv8)
-  Multi-person tracking (DeepSORT)
-  Deep feature extraction (ResNet50)
-  Person re-identification system
-  Live webcam processing
-  GPU acceleration (CUDA support)
-  Live metrics (FPS, active people, total count)
-  Interactive Streamlit dashboard
-  Historical tracking data
-  Feature fusion (deep + appearance features)
-  Modular AI architecture

---

## 🧠 AI Architecture

📷 Camera Input
↓
 YOLOv8 (Detection)
↓
 DeepSORT (Tracking)
↓
 ResNet50 (Feature Extraction)
↓
 Re-Identification Engine
↓
 Identity Matching System
↓
 Streamlit Dashboard



---

## 🛠 Tech Stack

| Layer | Technology |
|------|------------|
|  Detection | YOLOv8 |
|  Tracking | DeepSORT |
|  Features | ResNet50 |
|  ML Framework | PyTorch |
|  CV | OpenCV |
|  UI | Streamlit |
|  Data | NumPy, Pandas |
|  Similarity | Cosine Similarity, Histogram Matching |

---

## ⚙️ Installation

bash
git clone https://github.com/Hamze-alseed/People-Detection-Re-identification.git
cd People-Detection-Re-identification
pip install -r requirements.txt

## ▶️ Run the System's UI

streamlit run app.py

## 📊 Live Metrics

The system provides:

 FPS monitoring

 Active tracked persons

 Total unique persons

 Device type (CPU / GPU)

 Detection confidence filtering

 Identity stability tracking

## 🧬 Re-Identification Strategy

This system uses feature fusion:

 Deep visual embeddings (ResNet50)

 Clothing color histograms

 Temporal feature averaging

 Cosine similarity matching

 Threshold-based identity validation

 Memory-based identity persistence

## This allows stable identity tracking even with:

 Occlusions

 Re-entries

 Temporary disappearances

 Camera motion

 Lighting changes

## 🎯 Use Cases

 Smart surveillance systems

 Retail analytics

 Crowd monitoring

 Smart buildings

 Access control systems

 Campus security

 CV research

 AI system prototyping

 Real-time ML deployment

## 🧪 Research Value

This project demonstrates:

 Multi-model AI pipelines

 Real-time inference engineering

 Model fusion strategies

 Tracking + ReID integration

 System optimization

 GPU acceleration

 Production-style AI architecture

 End-to-end AI system design

## 🚀 Future Extensions

 Multi-camera support

 RTSP stream input

 Database integration

 REST API (FastAPI)

 Web dashboard (React)

 Cloud deployment

 Kubernetes scaling

 Face recognition module

 Behavior analysis

 Zone-based analytics

 Alert system

 Access control integration

## 📸 Demo
![System Demo](assets/output_test_video.gif)


## 👤 Author

Hamze Alseed
 AI Engineer |  Computer Vision |  Deep Learning |  Real-Time AI Systems

GitHub: https://github.com/Hamze-alseed
