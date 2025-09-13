# Hand Helm - Real-time Hand Gesture Recognition System

A computer vision project that uses hand gestures to control your computer, built with OpenCV, MediaPipe, and machine learning.

## 🎯 Features

- **Real-time Hand Detection**: Uses MediaPipe for robust hand landmark detection
- **Gesture Recognition**: Machine learning models to classify hand gestures
- **Computer Control**: Execute actions like media control, presentation navigation, and cursor control
- **Customizable Actions**: Map gestures to custom computer actions
- **High Performance**: Optimized for real-time processing

## 🎮 Supported Gestures

- Fist
- Open Palm
- Thumbs Up
- Thumbs Down
- Peace Sign
- OK Sign
- Point Up
- Point Down
- Point Left
- Point Right

## 🛠️ Tech Stack

- **Computer Vision**: OpenCV, MediaPipe
- **Machine Learning**: TensorFlow, scikit-learn
- **Data Processing**: NumPy, Pandas
- **System Control**: pyautogui, keyboard, pynput
- **Visualization**: Matplotlib, Seaborn

## 🚀 Quick Start

1. **Clone the repository**:
```bash
git clone https://github.com/AdityaKuchhal/HandHelm.git
cd HandHelm
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Run the application**:
```bash
python run.py
```

## 📁 Project Structure

```
hand-helm/
├── data/
│   ├── raw/           # Raw gesture data (gitignored)
│   └── processed/     # Preprocessed data (gitignored)
├── models/            # Trained models (gitignored)
├── src/
│   ├── config.py      # Configuration settings
│   ├── data_collection.py
│   ├── data_preprocessing.py
│   ├── gesture_detection.py
│   ├── gesture_classification.py
│   ├── computer_control.py
│   └── app.py         # Main application
├── scripts/
│   ├── data_management.py
│   ├── demo_data_collection.py
│   └── test_data_collection.py
├── tests/             # Unit tests
├── run.py             # Main launcher
└── requirements.txt   # Dependencies
```

## 🎯 Current Status

**Phase 1-2 Complete**: Project foundation and data collection system
- ✅ Basic hand detection with MediaPipe
- ✅ Data collection pipeline
- ✅ Data preprocessing system
- ✅ Machine learning model architecture
- ✅ Computer control actions
- ✅ Main application framework

**Next Steps**: 
- 🔄 Collect real gesture data
- 🔄 Train models on real data
- 🔄 Improve gesture recognition accuracy
- 🔄 Add more gesture types
- 🔄 Enhance user interface

## 🎮 Usage

### Basic Usage
```bash
python run.py
```

### Data Collection
```bash
python src/data_collection.py
```

### Testing
```bash
python scripts/test_data_collection.py
```

## ⚙️ Configuration

Edit `src/config.py` to customize:
- Gesture recognition settings
- Model parameters
- Action mappings
- Camera settings

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- MediaPipe for hand detection
- OpenCV for computer vision
- TensorFlow for machine learning

## 📧 Contact

For questions or feedback, please open an issue or contact me directly at adityakuchhal76@gmail.com.

Enjoy your journey towards touchless computing with Hand Helm!