# Neuro Pointer

**Neuro Pointer** is a real-time AI virtual mouse for Windows laptops that allows you to control the mouse cursor using **hand gestures or facial movement** through a webcam.

The system uses **MediaPipe + OpenCV** to track hand landmarks and facial landmarks and converts them into mouse actions.

The project supports two runtime control modes:

1. **Hand Gesture Mode** – Control the cursor using hand movements and gestures.
2. **Face Control Mode** – Control the cursor using **nose movement + eye blink detection**.

You can switch modes instantly using keyboard shortcuts.

---

# Core Technologies

- Python
- OpenCV
- MediaPipe (Hands + FaceMesh)
- PyAutoGUI
- NumPy

---

# Control Modes

## 1. Hand Gesture Mode

Uses **MediaPipe Hands** to detect hand landmarks and recognize gestures.

Features:

- Cursor movement using **index finger tip**
- **Left click** using thumb-index pinch
- **Right click** using index-middle touch
- **Scroll** using two-finger vertical movement
- **Drag** using closed fist gesture
- Optional **system volume control**

Hand mode is designed for **high accuracy cursor control**.

---

## 2. Face Control Mode (Eye Mode)

Uses **MediaPipe FaceMesh** to track facial landmarks.

Cursor movement is primarily controlled by **nose position**, with optional eye gaze micro-adjustments.

Pipeline:

```
FaceMesh
↓
Nose landmark tracking
↓
Cursor mapping
↓
Blink detection for click
```

Features:

- Cursor movement using **nose direction**
- **Blink detection** for left click
- **Adaptive smoothing** for stable cursor movement
- **Dead-zone center** to reduce jitter
- Optional **gaze micro adjustment**

This design significantly improves stability compared to pure gaze tracking.

---

# Hybrid Cursor Control System

The cursor control logic combines multiple techniques:

```
FaceMesh tracking
↓
Nose movement (primary cursor driver)
↓
Optional gaze micro adjustment
↓
Dead-zone filtering
↓
Adaptive smoothing
↓
Cursor movement
```

Benefits:

- Reduced jitter
- Less head movement required
- Smooth cursor motion
- More natural control

---

# Cursor Re-Center Feature

A dedicated **recenter shortcut** is included.

```
X key → Move cursor to screen center
```

Behavior:

```
Press X
↓
Cursor moves to screen center
↓
Face center recalibrates
↓
Tracking resumes smoothly
```

This prevents cursor drift during long usage.

---

# Real-Time Performance

The application runs at:

```
20 – 30 FPS
```

To maintain performance:

- Only one pipeline runs at a time
- Either **hand tracking** OR **face tracking**

---

# Project Structure

```
neuro_pointer/
│
├── main.py
├── config.py
├── utils.py
│
├── hand_tracking/
│   ├── hand_detector.py
│   └── gesture_recognition.py
│
├── eye_tracking/
│   ├── eye_detector.py
│   └── gaze_tracking.py
│
├── controllers/
│   ├── mouse_controller.py
│   └── volume_controller.py
│
├── calibration/
│   ├── calibrate.py
│   └── calibration_data.json
│
├── ui/
│   └── overlay.py
│
├── requirements.txt
└── README.md
```

---

# Installation (Windows)

Open PowerShell inside the project folder.

### 1. Create virtual environment

```
python -m venv .venv
```

### 2. Activate environment

```
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```
pip install -r requirements.txt
```

---

# Run the Application

```
python main.py
```

---

# Keyboard Controls

| Key     | Function                         |
| ------- | -------------------------------- |
| H       | Switch to Hand Mode              |
| E       | Switch to Eye / Face Mode        |
| T       | Start / Stop training            |
| SPACE   | Capture training sample          |
| C       | Calibrate face center            |
| X       | Recenter cursor to screen center |
| R       | Reset calibration values         |
| Q / ESC | Quit application                 |

---

# Gesture Training System

Training allows the system to adjust gesture thresholds based on your usage.

Steps:

1. Press **T** to start training.
2. Follow the on-screen instruction.
3. Press **SPACE** to capture gesture samples.
4. Training automatically saves values to:

```
calibration/calibration_data.json
```

---

# Camera Configuration

Recommended webcam settings:

```
Resolution : 960 × 540
FPS        : 30
```

Higher resolution may reduce performance.

---

# Performance Tips

For best results:

- Use **good lighting**
- Keep webcam **stable**
- Sit **60-80 cm** away from camera
- Avoid strong backlight

---

# Troubleshooting

### Cursor jitter

Increase smoothing values inside:

```
config.py
```

---

### Blink click triggers too often

Run **eye training** and capture better samples.

---

### Cursor requires too much head movement

Adjust control region inside:

```
eye_tracking/gaze_tracking.py
```

---

### Webcam not detected

Check:

```
config.CAMERA_INDEX
```

---

# Future Improvements

Possible upgrades:

- Eye-gesture click
- Multi-monitor support
- Auto sensitivity calibration
- Head-pose based cursor control
- Accessibility mode for disabled users

---

# License

This project is intended for **educational and experimental purposes**.

---

# Author

**Neuro Pointer Project**

AI-based webcam mouse controller built with Python and MediaPipe.
