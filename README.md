# 🕵️‍♂️ People Detection & Re-Identification System

> Real-time multi-person detection, tracking, and re-identification system built for real-world AI applications using deep learning, computer vision, and real-time inference pipelines.

---

## 🚀 Overview

This project is a **real-time AI system** that performs:
- 👁️ Human detection  
- 🧭 Multi-object tracking  
- 🧠 Person re-identification  
- 🧬 Feature-based identity matching  
- 🎥 Real-time visualization  
- ⚡ GPU-accelerated inference  

Built as a **full AI pipeline**, not just a model demo.

---

## 🎯 Key Features

- 🔍 Real-time person detection (YOLOv8)
- 🧭 Multi-person tracking (DeepSORT)
- 🧠 Deep feature extraction (ResNet50)
- 🧬 Person re-identification system
- 🎥 Live webcam processing
- ⚡ GPU acceleration (CUDA support)
- 📊 Live metrics (FPS, active people, total count)
- 🖥 Interactive Streamlit dashboard
- 🗂 Historical tracking data
- 🔄 Feature fusion (deep + appearance features)
- 🧩 Modular AI architecture

---

## 🧠 AI Architecture

📷 Camera Input
↓
🧠 YOLOv8 (Detection)
↓
🧭 DeepSORT (Tracking)
↓
🧬 ResNet50 (Feature Extraction)
↓
🧠 Re-Identification Engine
↓
🧬 Identity Matching System
↓
🖥 Streamlit Dashboard



---

## 🛠 Tech Stack

| Layer | Technology |
|------|------------|
| 🎯 Detection | YOLOv8 |
| 🧭 Tracking | DeepSORT |
| 🧠 Features | ResNet50 |
| 🔥 ML Framework | PyTorch |
| 👁️ CV | OpenCV |
| 🖥 UI | Streamlit |
| 📊 Data | NumPy, Pandas |
| 🧬 Similarity | Cosine Similarity, Histogram Matching |

---

## ⚙️ Installation

bash
git clone https://github.com/Hamze-alseed/People-Detection-Re-identification.git
cd People-Detection-Re-identification
pip install -r requirements.txt

## ▶️ Run the System's UI

streamlit run app.py

## The system provides:

⚡ FPS monitoring

👥 Active tracked persons

🧮 Total unique persons

💻 Device type (CPU / GPU)

🎯 Detection confidence filtering

🧠 Identity stability tracking
