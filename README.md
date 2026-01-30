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
