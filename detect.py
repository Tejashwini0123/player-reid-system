import cv2
import numpy as np
import torch
from ultralytics import YOLO
from torchvision.models import resnet50
import torchvision.transforms as transforms
import torch.nn as nn
import os

class PlayerReIDSystem:
    def __init__(self, model_path, video_path):
        self.model = YOLO(model_path)
        self.video_path = video_path
        self.cap = None
        self.is_playing = False
        self.current_frame = 0
        self.total_frames = 0

        self.next_id = 1
        self.active_tracks = {}
        self.disappeared_tracks = {}
        self.max_disappeared = 30
        self.similarity_threshold = 0.6
        self.colors = self.generate_colors(100)

        self.feature_extractor = resnet50(pretrained=True)
        self.feature_extractor.fc = nn.Identity()
        self.feature_extractor.eval()

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def generate_colors(self, num):
        np.random.seed(42)
        return [tuple(np.random.randint(0, 255, 3).tolist()) for _ in range(num)]

    def extract_features(self, image, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return np.zeros(2048)
        try:
            tensor = self.transform(crop).unsqueeze(0)
            with torch.no_grad():
                feature = self.feature_extractor(tensor).squeeze().numpy()
            return feature / (np.linalg.norm(feature) + 1e-6)
        except:
            return np.zeros(2048)

    def calculate_similarity(self, feat1, feat2):
        return np.dot(feat1, feat2) / (np.linalg.norm(feat1) * np.linalg.norm(feat2) + 1e-6)

    def calculate_iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

    def process_video(self):
        cap = cv2.VideoCapture(self.video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        out = cv2.VideoWriter("output_with_reid.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = self.model(frame, verbose=False)
            detections = []
            for result in results:
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confs = result.boxes.conf.cpu().numpy()
                    for box, conf in zip(boxes, confs):
                        if conf > 0.5:
                            feat = self.extract_features(frame, box)
                            detections.append((box, conf, feat))

            matched_ids = self.match_detections(detections, frame_idx)

            for i, (box, _, _) in enumerate(detections):
                if i < len(matched_ids):
                    tid = matched_ids[i]
                    color = self.active_tracks[tid]['color']
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"ID: {tid}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            out.write(frame)
            print(f"Processing frame {frame_idx+1}/{total_frames}", end="\r")
            frame_idx += 1

        cap.release()
        out.release()
        print("\nProcessing complete. Video saved as 'output_with_reid2.mp4'.")

    def match_detections(self, detections, frame_idx):
        matched_ids = []
        unmatched = []

        for box, conf, feat in detections:
            best_id = None
            best_sim = 0

            for tid, info in self.active_tracks.items():
                sim = self.calculate_similarity(feat, info['features'])
                iou = self.calculate_iou(box, info['bbox'])
                score = 0.7 * sim + 0.3 * iou
                if score > best_sim and score > self.similarity_threshold:
                    best_sim = score
                    best_id = tid

            if best_id is not None:
                self.active_tracks[best_id]['bbox'] = box
                self.active_tracks[best_id]['features'] = 0.8 * self.active_tracks[best_id]['features'] + 0.2 * feat
                matched_ids.append(best_id)
            else:
                unmatched.append((box, feat))

        for box, feat in unmatched:
            tid = self.next_id
            self.next_id += 1
            self.active_tracks[tid] = {'bbox': box, 'features': feat, 'color': self.colors[tid % len(self.colors)]}
            matched_ids.append(tid)

        return matched_ids

if __name__ == "__main__":
    MODEL_PATH = "yolov11_model.pt" 
    VIDEO_PATH = "15sec_input_720p.mp4"

    app = PlayerReIDSystem(model_path=MODEL_PATH, video_path=VIDEO_PATH)
    app.process_video()
