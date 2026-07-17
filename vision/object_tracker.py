import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import numpy as np
from ultralytics import YOLO

class SubwayObjectTracker:
    def __init__(self, model_path=r"runs\detect\models\custom_subway_yolo-5\weights\best.onnx"):
        # We start with the custom fine-tuned exported model
        self.model = YOLO(model_path, task="detect")
        
    def track_frame(self, frame):
        # Use standard detection (stateless) instead of tracking to prevent memory leaks over 100k steps
        results = self.model.predict(frame, verbose=False)
        self.last_results = results
        
        tracked_entities = []
        if not results or not results[0].boxes:
            return tracked_entities
            
        boxes = results[0].boxes
        
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            class_id = int(box.cls[0])
            
            
            # Since we are not tracking across frames anymore, ID is not needed
            track_id = -1
            
            tracked_entities.append({
                "id": track_id,
                "class": class_id,
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "center_x": int((x1 + x2) / 2),
                "center_y": int((y1 + y2) / 2)
            })
            
        return tracked_entities
