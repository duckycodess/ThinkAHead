#!/usr/bin/env python3

from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional


class Detection:
    """Represents a single detection"""
    
    CLASS_NAMES = ['motorcycle', 'rider', 'helmet', 'no_helmet', 'license_plate']
    
    def __init__(self, bbox: List[float], class_id: int, confidence: float):
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.class_id = class_id
        self.class_name = self.CLASS_NAMES[class_id] if class_id < len(self.CLASS_NAMES) else f'class_{class_id}'
        self.confidence = confidence
        
        # Computed properties
        self.x1, self.y1, self.x2, self.y2 = bbox
        self.width = self.x2 - self.x1
        self.height = self.y2 - self.y1
        self.center_x = (self.x1 + self.x2) / 2
        self.center_y = (self.y1 + self.y2) / 2
        self.area = self.width * self.height
    
    def __repr__(self):
        return f"Detection({self.class_name}, conf={self.confidence:.2f})"
    
    def iou(self, other: 'Detection') -> float:
        """Calculate Intersection over Union with another detection"""
        x1 = max(self.x1, other.x1)
        y1 = max(self.y1, other.y1)
        x2 = min(self.x2, other.x2)
        y2 = min(self.y2, other.y2)
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        union = self.area + other.area - intersection
        
        return intersection / union if union > 0 else 0.0


class YOLODetector:
    """YOLO model wrapper for ThinkAHead"""
    
    def __init__(self, model_path: str = 'models/trained/thinkahead_best.pt'):
        """Load the YOLO model"""
        self.model_path = Path(model_path)
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        print(f"Loading model from {model_path}...")
        self.model = YOLO(str(model_path))
        print("Model loaded successfully!")
    
    def predict(
        self,
        frame: np.ndarray,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45
    ) -> List[Detection]:

        results = self.model.predict(
            frame,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False
        )[0]
        
        detections = []
        
        if results.boxes is not None and len(results.boxes) > 0:
            boxes = results.boxes.xyxy.cpu().numpy()
            confidences = results.boxes.conf.cpu().numpy()
            class_ids = results.boxes.cls.cpu().numpy().astype(int)
            
            for bbox, conf, cls_id in zip(boxes, confidences, class_ids):
                det = Detection(
                    bbox=bbox.tolist(),
                    class_id=int(cls_id),
                    confidence=float(conf)
                )
                detections.append(det)
        
        return detections


_model: Optional[YOLODetector] = None


def load_model(model_path: str = 'models/trained/thinkahead_best.pt') -> YOLODetector:
    """Load or return cached model"""
    global _model
    if _model is None:
        _model = YOLODetector(model_path)
    return _model


def get_model() -> Optional[YOLODetector]:
    """Get the loaded model (None if not loaded)"""
    return _model


# Test
if __name__ == "__main__":
    # Test loading model
    model = load_model()
    
    # Test on a dummy image
    dummy = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    detections = model.predict(dummy)
    print(f"Detections on random image: {len(detections)}")
    
    # Test on a real image if available
    test_images = list(Path('data/processed/images/test').glob('*.jpg'))[:1]
    if test_images:
        img = cv2.imread(str(test_images[0]))
        detections = model.predict(img)
        print(f"\nDetections on {test_images[0].name}:")
        for det in detections:
            print(f"  {det}")