Player Re-Identification System using YOLOv11 + ResNet50
This project implements an advanced Player Re-Identification (Re-ID) pipeline using:

YOLOv11 for person detection

ResNet50 for appearance-based feature extraction

Cosine similarity and IoU for identity matching and tracking

The system reads an input video, detects players frame-by-frame, extracts features for each player, and tracks consistent identities across frames with assigned unique IDs.

📁 Project Structure
bash
Copy
Edit
├── reid_system.py           # Main script with PlayerReIDSystem class
├── yolov11_model.pt         # Pre-trained YOLOv11 weights (your own)
├── 15sec_input_720p.mp4     # Input video to test the system
├── output_with_reid.mp4     # Output video with bounding boxes and IDs
├── README.md                # You're here
🚀 Features
✅ Detects players using YOLOv11
✅ Extracts visual features using ResNet50
✅ Matches identities across frames using similarity + IoU
✅ Assigns and maintains consistent ID colors
✅ Outputs annotated video as output_with_reid.mp4

🔧 Requirements
Install dependencies using:

bash
Copy
Edit
pip install -r requirements.txt
If you don't have a requirements.txt, here's what you need:

bash
Copy
Edit
pip install torch torchvision opencv-python ultralytics numpy
🧪 How to Run
Ensure your .pt model and input .mp4 file exist in the directory. Then run:

bash
Copy
Edit
python reid_system.py
✅ Output video will be saved as output_with_reid.mp4 in the same directory.

📦 Model & Input Paths
Edit the following lines in reid_system.py with your actual files:

python
Copy
Edit
MODEL_PATH = "yolov11_model.pt"
VIDEO_PATH = "15sec_input_720p.mp4"
🧠 How It Works
YOLOv11 detects all people in each frame.

ResNet50 extracts a 2048-dimensional feature vector for each person.

Features are compared with previously tracked individuals using cosine similarity + IoU.

New players are assigned a new ID; existing ones are tracked using their ID and color.
