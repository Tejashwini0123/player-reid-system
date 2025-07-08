# Player Re-Identification System 🎯

This project implements an advanced **Player Re-Identification (Re-ID)** system using **YOLOv11** for object detection and **ResNet50** for appearance-based feature extraction. The goal is to track and consistently identify players across video frames based on visual similarity.

---

## 📁 Project Structure

player-reid-system/
├── player_reid_system.py # Main Python script
├── yolov11_model.pt # YOLOv11 model weights (replace with actual model)
├── 15sec_input_720p.mp4 # Sample input video
├── requirements.txt # All dependencies
├── README.md # This file

---

## 🚀 How to Run

### 1. Clone the repository:
git clone https://github.com/Tejashwini0123/player-reid-system.git
cd player-reid-system
2. Set up a virtual environment (optional but recommended):
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
3. Install dependencies:
pip install -r requirements.txt
4. Place your video and YOLO model in the folder, then run:
python detect.py
The output video will be saved as: output_with_reid.mp4

🧠 How It Works
Detection: Uses YOLOv11 to detect players (bounding boxes).

Feature Extraction: Uses a pretrained ResNet50 model with the FC layer removed to extract 2048-dimensional features from each detected player.

Matching: Computes cosine similarity + IoU score between new detections and existing tracks to maintain consistent IDs.

Visualization: Draws bounding boxes and player IDs on each frame and saves the processed video.

⚙️ Dependencies
All required packages are listed in requirements.txt. Key ones include:
torch
ultralytics
opencv-python
numpy
torchvision
Pillow
Make sure you have Python 3.8+ installed.
🧾 Notes
Confidence threshold is set to 0.5 for detections.
You can adjust the similarity and IoU weights in the code for different scenarios.
Designed to be simple, modular, and easy to extend (e.g., DeepSORT, motion models).

🙋‍♀️ Author
Tejashwini K
Email: [tejashwinichary67.com]
GitHub: github.com/Tejashwini0123
