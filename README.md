# Real-Time Chess Vision & Analysis Pipeline ♟️🤖
<img width="1871" height="932" alt="tahta+openCV" src="https://github.com/user-attachments/assets/cb358663-66b3-404a-bd37-04cf80d2ae57" />

<img width="4096" height="2304" alt="kenditahtam (1)" src="https://github.com/user-attachments/assets/b64fea1a-a257-4113-a608-6f713e9cc214" />

<img width="1126" height="793" alt="pygameanalyze" src="https://github.com/user-attachments/assets/8d855118-5715-4352-ac66-49a1ccaf37e3" />

<br>
<br>

An end-to-end Computer Vision and AI system that detects chess positions from a live camera feed, converts them into FEN notation, and analyzes the board using the Stockfish chess engine.

The project combines **YOLOv8 object detection**, **OpenCV perspective transformation**, **interactive board correction**, and **engine-based move analysis** into a single real-time pipeline.

## 💡 Motivation & The Development Journey

My primary goal for this project was born out of a personal need: I realized there are almost no reliable, completely offline applications on the market that allow users to snap a picture of a physical chessboard, convert it to FEN format, and instantly analyze it with a chess engine. I wanted to build exactly that.

Initially, my vision was fully automated: the system would use a phone camera, auto-detect the board corners, read the pieces, and feed the FEN directly to Stockfish. However, real-world computer vision is challenging, and the journey didn't go exactly as planned:

* **The Object Detection Struggle:** I quickly noticed that pre-trained YOLO models struggled heavily with recognizing chess pieces from different angles. To fix this, I photographed my own physical chessboard and trained a custom YOLO model on Roboflow. I achieved an 86.4% accuracy rate, but for chess, a single wrong piece ruins the entire analysis. Eventually, I had to revert to a more generalized model I found online.
* **The AI API Attempt:** Hoping for better accuracy, I integrated the Gemini API to analyze the board images and generate the FEN. Unfortunately, LLMs still struggle with precise spatial reasoning for chessboards. More importantly, using an API required an active internet connection, which violated my core goal of keeping this tool 100% offline. I decided to remove it.
* **The Pivot to UI (Human-in-the-Loop):** Since the piece recognition performance was still too low for full automation, I pivoted my approach. I decided to build a "human-in-the-loop" system. I first built a user interface using Tkinter to show the detected board and let the user correct the AI's mistakes manually. I wasn't satisfied with Tkinter's look and feel, so I completely rebuilt the interface using **Pygame** to make the drag-and-drop experience smoother.

While the final piece recognition accuracy didn't meet my initial ambitious expectations, this project became an incredible engineering journey. I learned how to train custom models, integrate external engines, build interactive GUIs, and most importantly, how to pivot and find practical workarounds when hitting technical limits. I am sharing this project on GitHub exactly as it is—imperfections included—as a foundation for myself and others to improve upon.

# 🚀 Features

### ♟️ Real-Time Chess Piece Detection

* Custom-trained YOLOv8 model for detecting and classifying chess pieces in real time.
* Supports webcam or IP camera streams.

### 📐 Perspective Transformation & Board Mapping

* Uses OpenCV perspective warping to detect and flatten the chessboard.
* Converts physical board coordinates into a logical 8×8 matrix.

### 🧠 AI Hallucination Filtering

* Automatically removes impossible detections using rule-based chess validation.
* Example: prevents pawns from appearing on the 1st or 8th rank.

### 🖱️ Human-in-the-Loop Correction Interface

* Interactive drag-and-drop Pygame GUI.
* Allows users to manually fix incorrect detections before analysis.

### 🤖 Chess Engine Integration

* Generates valid FEN notation from the detected board.
* Uses Stockfish to calculate the best move from the current position.

### 📦 Automatic Chess Asset Downloader

* Automatically downloads high-quality chess piece sprites from Chess.com if assets are missing.

---

# 🛠️ Installation & Setup

## 1. Clone the Repository

```bash
git clone https://github.com/ZeroHighx/ChessPieces_detection_with_camera_and_YOLOv8.git
cd ChessPieces_detection_with_camera_and_YOLOv8
```

---

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 3. Download the YOLO Model

The trained model file:

```bash
chess-model-yolov8m.pt
```

is available in the **Releases** section of this repository.

Download it and place it in the project root directory.

---

## 4. Download Stockfish

Download the Stockfish engine from:

https://stockfishchess.org/download/

Then update the path inside the script:

```python
STOCKFISH_PATH = r"C:\path\to\stockfish.exe"
```

---

## 5. Configure Camera Source

For default webcam:

```python
CAMERA_ADDRESS = 0
```

For IP webcam:

```python
CAMERA_ADDRESS = "http://YOUR_PHONE_IP:8080/video"
```

---

## 6. Run the Project

```bash
python chess_camera_detection.py
```

---

# 🎮 Controls

| Key | Action                           |
| --- | -------------------------------- |
| A   | Automatic board detection        |
| R   | Reset selected corner points     |
| C   | Capture frame and start analysis |
| Q   | Quit application                 |

---

# 🔄 Pipeline Overview

```text
Camera Feed
     ↓
Board Detection
     ↓
Perspective Warp
     ↓
YOLOv8 Piece Detection
     ↓
Rule-Based Chess Validation
     ↓
Manual GUI Correction
     ↓
FEN Generation
     ↓
Stockfish Analysis
     ↓
Best Move Recommendation
```

---

# 🧰 Technologies Used

* Python
* OpenCV
* YOLOv8 (Ultralytics)
* NumPy
* Pygame
* Stockfish Engine

---

# 📌 Future Improvements

* Full automatic move tracking
* Chess clock integration
* Online multiplayer board recognition
* PGN export support
* Mobile application version
* Stronger board auto-detection system

---

# 📄 License

This project is licensed under the MIT License.

