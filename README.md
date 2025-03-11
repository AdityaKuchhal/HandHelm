# Hand Helm

Hand Helm is a real-time hand gesture recognition system that enables touchless control of your computer. By leveraging computer vision and machine learning, Hand Helm interprets hand gestures captured from your webcam and maps them to various computer functions such as presentation navigation, media playback, cursor control, and more.

## Overview

Hand Helm offers a seamless and intuitive way to interact with your computer without physical contact. It utilizes MediaPipe for hand detection and landmark extraction, combined with a machine learning model built with TensorFlow or PyTorch for gesture classification. The project is designed with modularity and reusability in mind, ensuring clean code structure and easy maintainability.

## Features

- **Real-Time Gesture Recognition:**  
  Capture and process hand gestures in real time using your webcam.
- **Customizable Gesture Mapping:**  
  Easily assign specific gestures to control functions like slide navigation, media playback, cursor movement, and more.
- **Visual Feedback:**  
  Overlay hand landmarks and recognized gesture labels on the video feed for immediate user feedback.
- **Optimized & Modular Codebase:**  
  Designed with separate modules for gesture detection, classification, and utilities to ensure ease of maintenance and future expansion.

## Tech Stack

- **Programming Language:** Python 3.8+
- **Computer Vision:** OpenCV, MediaPipe
- **Machine Learning:** TensorFlow or PyTorch (choose based on your preference)
- **Data Processing:** NumPy, Pandas
- **Visualization:** Matplotlib; optionally Streamlit or Tkinter for a graphical user interface
- **Version Control:** Git

## Recommended PC Specifications

- **Operating System:** Windows 10/11, Linux, or macOS
- **CPU:** Modern multi-core processor (e.g., Intel i5/i7 or AMD Ryzen 5/7)
- **RAM:** Minimum 8GB (16GB recommended)
- **GPU:** NVIDIA GPU with CUDA support (e.g., GTX 1060 or better) for accelerated model training/inference (optional)
- **Webcam:** 720p resolution or higher
- **Storage:** SSD with sufficient space for datasets, models, and project files

## Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/AdityaKuchhal/HandHelm.git
   cd hand_helm
   ```
2. **Install the Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

# Usage

1. **Data Preparation:**
   If using your own dataset, place your raw images or videos in the designated data folder and use the provided scripts to process and augment your data.
2. **Training the Model:**
   Configure hyperparameters and settings in the configuration file, then run:
   ```bash
   python src/gesture_classification.py --train
   ```
3. **Running the Application:**
   Launch the application to see real-time gesture recognition:
   ```bash
    python src/app.py
   ```
4. **Customizing Gesture Actions:**
   Modify the gesture-to-action mappings in the main application file to suit your needs.

# Contributing

Contributions are welcome! If you have improvements, new features, or bug fixes, please submit a pull request or open an issue.

# License

This project is licensed under the MIT License. See the LICENSE file for details.

# Contact

For questions or feedback, please open an issue or contact me directly at adityakuchhal76@gmail.com.

Enjoy your journey towards touchless computing with Hand Helm!
