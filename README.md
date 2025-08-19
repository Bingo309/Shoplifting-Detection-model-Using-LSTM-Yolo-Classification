# VigiGuard: Secure AI for Real-Time Theft Detection

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2-%23EE4C2C)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.10-%235C3EE8)](https://opencv.org/)

This repository contains the core machine learning models for **VigiGuard**, an AI-powered system designed to detect shoplifting behavior in real-time from surveillance footage. The project was developed as a Bachelor's thesis in Computer Science.

**Disclaimer:** This repo contains only the ML model component. The full-stack web application (frontend, backend, database) is in a private repository.

![VigiGuard System Overview](images/system_architecture.png) *(Use Figure 1.4 from your doc)*

## 🧠 How It Works: The ML Pipeline

The system employs a multi-stage pipeline to accurately identify theft behavior:

1.  **Person Detection:** A YOLOv8 model detects and tracks individuals in the video frame.
2.  **Pose Estimation:** YOLOv11-m-pose extracts 17 body keypoints for each person.
3.  **Dual-Stream Analysis:**
    *   **Arm Pose Classification:** A Random Forest model classifies the arm's pose (`hand-on-shelf`, `hand-in-pocket`, `passive`) based on shoulder, elbow, and wrist keypoints.
    *   **Grasp Classification:** A cropped image of the hand (guided by the wrist keypoint) is classified as `grasping` or `not grasping` by a MobileNetV2 + SVM pipeline.
4.  **Temporal Context (LSTM):** An LSTM network analyzes sequences of poses and grasp states to recognize complex theft behaviors over time.
5.  **Decision Fusion:** A theft alert is triggered only when specific conditions are met simultaneously (e.g., `grasping` + `hand-in-pocket`).

![Grasp Classification Workflow](media/grasp_demo.gif) *(Create a GIF showing this process)*

## 📊 Model Performance & Results

### Grasp Classifier (MobileNetV2 + SVM)
- **Validation Accuracy:** **97.3%**
- The model generalizes well to various lighting conditions and backgrounds.

![Grasp Training Curves](images/grasp_training_curves.png) *(Use a panel from Figure 6-2)*

### Arm Pose Classifier (Random Forest)
- Achieved high accuracy in distinguishing critical poses.
- **Key Insight:** The biggest challenge was distinguishing between `hand-in-pocket` and `passive` poses when the hand was near the pocket.

![Pose Confusion Matrix](images/pose_confusion_matrix.png) *(Use Figure 6-5)*

### Full Pipeline
The integrated system successfully flags individuals only when the behavioral conditions for theft are met, significantly reducing false positives compared to single-frame analysis.

![Real-Time Theft Detection](media/real_time_demo.gif) *(A short video clip of your system working)*

## 🚀 Getting Started

### Prerequisites
*   Python 3.11+
*   CUDA-capable GPU (recommended)

### Installation
1.  Clone the repo:
    ```bash
    git clone https://github.com/your-username/VigiGuard-AI-Theft-Detection.git
    cd VigiGuard-AI-Theft-Detection
    ```
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Usage
1.  **Download Model Weights:** [Provide a link if you host them on Google Drive etc.]
2.  Run the inference demo on a video file:
    ```bash
    python src/3_inference/theft_detection_pipeline.py --input path/to/your/video.mp4
    ```

## 📁 Project Structure
*(Put the folder tree from Section 1 here)*

## 🛠️ Development Process
The project followed an iterative approach. Our key learnings were:
- **Attempt 1 (Failed):** Training on raw RGB frames led to overfitting and poor generalization.
- **Attempt 2 (Failed):** Using YOLO for direct "item-in-hand" detection was too noisy.
- **Attempt 3 (Success):** The wrist-keypoint-guided cropping method proved to be the most robust and efficient approach, enabling reliable grasp classification.

## 👨‍💻 Authors
- **Mohammed Abdulaziz Banafe**
- **Mustafa Muhamed Aleid**
- **Shaden Kherallah Althagafi**
- **Basil Hatim Alharbi**
- **Sara Essam Alhilali**

*Supervised by Dr. Zainab Mahmoud & Ms. Bushru Al-Anesi*

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments
We thank Jeddah International College and our advisors for their guidance and support.
