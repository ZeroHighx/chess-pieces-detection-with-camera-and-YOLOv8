# Real-Time Chess Vision & Analysis Pipeline ♟️🤖

This project is an end-to-end computer vision and AI pipeline that detects chess pieces from a live camera feed, generates FEN codes, and uses the Stockfish engine to recommend the best moves. It features a fault-tolerant hybrid architecture utilizing both local YOLO models and the cloud-based Google Gemini API.

## 🚀 Features
* **Real-Time Object Detection:** Custom trained YOLOv8 model for detecting and classifying chess pieces.
* **Geometric Mapping:** OpenCV perspective warping to track board corners and map physical coordinates to a logical 8x8 matrix.
* **Hybrid AI Fallback:** Integrates Google Gemini API (Multimodal VLM) as a backup for complex board states where local detection confidence is low.
* **Human-in-the-Loop GUI:** An interactive Tkinter-based interface allowing users to manually correct AI misclassifications before engine evaluation.
* **Chess Engine Integration:** Analyzes the validated board state using the Stockfish engine to predict the most optimal moves.

## 🛠️ Installation & Setup

1. Clone this repository:
   `git clone https://github.com/ZeroHighx/ChessPieces_detection_with_camera_and_YOLOv8.git`

2. Install the required dependencies:
   `pip install -r requirements.txt`

3. Download the [Stockfish Engine](https://stockfishchess.org/download/) and update the `STOCKFISH_PATH` variable in `camera_detected.py`.

4. Update the `CAMERA_ADDRESS` (use `0` for default webcam or your IP Camera URL).

5. Run the system:
   `python chess_camera_detection.py`

## 🎮 How to Use
* **A:** Detection Lock camera view and start automatic board tracking.
* **R:** Reset tracking points for manual corner selection (Click 4 corners of the board).
* **C:** Capture the current frame and run the Hybrid Analysis Pipeline (YOLO -> GUI Correction -> Stockfish).
