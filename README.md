# 👁️ Real-Time Eye Gaze Tracking and Attention Analysis System

## 📌 Overview

The Real-Time Eye Gaze Tracking and Attention Analysis System is an AI and Computer Vision based application that monitors user attention using a webcam.

The system uses MediaPipe Face Mesh for facial landmark detection and iris tracking, OpenCV for real-time video processing, and Streamlit for an interactive dashboard. It estimates gaze direction and analyzes user attention by classifying users as attentive or distracted.

The project also generates analytical visualizations such as attention timeline graphs, heatmaps, and session reports.

---

## 🚀 Features

* Real-time webcam monitoring
* Facial landmark detection
* Iris tracking
* Eye gaze estimation
* Attention classification
* Attention percentage calculation
* Distracted percentage calculation
* Attention timeline graph
* Attention heatmap visualization
* Interactive Streamlit dashboard
* Final session analytics report

---

## 🛠️ Technologies Used

### Programming Language

* Python 3.11

### Libraries and Frameworks

* OpenCV
* MediaPipe Face Mesh
* NumPy
* Matplotlib
* Streamlit

---

## ⚙️ System Workflow

1. Capture webcam frames using OpenCV
2. Detect facial landmarks using MediaPipe Face Mesh
3. Extract iris coordinates
4. Estimate gaze direction
5. Classify attention state
6. Generate analytics
7. Create timeline graph
8. Generate attention heatmap
9. Display results on Streamlit dashboard

---

## 📂 Project Structure

```text
Real-Time-Eye-Gaze-Tracking-Attention-Analysis/

│
├── app.py
├── requirements.txt
├── README.md
│
├── modules/
│   ├── face_detection.py
│   ├── eye_tracking.py
│   ├── attention_heatmap.py
│   └── attention_timeline.py
│
├── outputs/
│   ├── heatmap.png
│   ├── timeline.png
│   └── reports/
│
└── assets/
    └── screenshots/
```

---

## 📊 Results

### Initial Dashboard

<img width="1600" height="1000" alt="image" src="https://github.com/user-attachments/assets/89d16fa4-f18c-41b3-a7e9-48292a99ef2d" />

---

### Real-Time Eye Gaze Tracking

<img width="1600" height="1000" alt="image" src="https://github.com/user-attachments/assets/c7b00d6a-c544-48a5-a3d1-8f1b261ca3d2" />

<img width="1600" height="1000" alt="image" src="https://github.com/user-attachments/assets/bb936ba1-482c-4201-b724-eb2f4f4f088a" />

---

### Attention Timeline Graph

<img width="1600" height="1000" alt="image" src="https://github.com/user-attachments/assets/7572d002-ba22-40d9-be1f-97d95798617f" />


---

### Attention Heatmap

<img width="1600" height="1000" alt="image" src="https://github.com/user-attachments/assets/e7e30f93-c5d6-40f6-9a13-6430819d498c" />


---

### Analytics Dashboard

<img width="1600" height="1000" alt="image" src="https://github.com/user-attachments/assets/d10b1d7f-f610-4a62-bbcc-831441ca5afb" />


---

### Final Session Report

<img width="1600" height="1000" alt="image" src="https://github.com/user-attachments/assets/26dd65ba-6d02-4bb8-b077-d9aa14696edd" />


---

## ▶️ How to Run

Clone the repository:

```bash
git clone https://github.com/your-username/Real-Time-Eye-Gaze-Tracking-Attention-Analysis.git
```

Move into the project directory:

```bash
cd Real-Time-Eye-Gaze-Tracking-Attention-Analysis
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

## 📈 Applications

* Smart Classrooms
* Online Learning Platforms
* Examination Monitoring
* Productivity Tracking
* Human Computer Interaction
* User Behavior Analysis

---

## 🔮 Future Scope

* Multi-user tracking
* Emotion recognition
* Deep learning based attention prediction
* Mobile application integration
* Cloud analytics dashboard

---

## 📄 License

This project was developed for academic and educational purposes.

```
```
