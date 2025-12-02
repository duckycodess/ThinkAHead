#!/usr/bin/env python3
"""
ocr_utils.py - License plate OCR for ThinkAHead
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional
import re

# Try to import EasyOCR, fall back gracefully if not available
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    print("Warning: EasyOCR not installed. Run: pip install easyocr")


class PlateOCR:
    """License plate OCR using EasyOCR"""
    
    def __init__(self, languages: List[str] = ['en'], use_gpu: bool = True):
        if not EASYOCR_AVAILABLE:
            raise ImportError("EasyOCR is required. Install with: pip install easyocr")
        
        print("Initializing EasyOCR (first run downloads models)...")
        self.reader = easyocr.Reader(languages, gpu=use_gpu, verbose=False)
        print("EasyOCR ready!")
    
    def preprocess_plate(self, plate_img: np.ndarray) -> np.ndarray:
        """Preprocess plate image for better OCR"""
        if plate_img is None or plate_img.size == 0:
            return plate_img
        
        # Convert to grayscale
        if len(plate_img.shape) == 3:
            gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_img.copy()
        
        # Resize to standard height
        target_height = 80
        h, w = gray.shape
        if h > 0:
            scale = target_height / h
            new_w = int(w * scale)
            gray = cv2.resize(gray, (new_w, target_height), interpolation=cv2.INTER_CUBIC)
        
        # Denoise
        gray = cv2.bilateralFilter(gray, 11, 17, 17)
        
        # Adaptive threshold
        thresh = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11, 2
        )
        
        return thresh
    
    def read_plate(self, plate_img: np.ndarray, preprocess: bool = True) -> Tuple[str, float]:
        """
        Read text from license plate image.
        
        Returns:
            Tuple of (plate_text, confidence)
        """
        if plate_img is None or plate_img.size == 0:
            return "", 0.0
        
        # Try with preprocessing
        if preprocess:
            processed = self.preprocess_plate(plate_img)
            results = self.reader.readtext(processed, detail=1)
        
        # If no results, try original
        if not results:
            results = self.reader.readtext(plate_img, detail=1)
        
        if not results:
            return "", 0.0
        
        # Combine all text
        texts = []
        confidences = []
        
        for (bbox, text, conf) in results:
            clean = self._clean_text(text)
            if clean:
                texts.append(clean)
                confidences.append(conf)
        
        if not texts:
            return "", 0.0
        
        combined = ''.join(texts).upper()
        avg_conf = sum(confidences) / len(confidences)
        
        return combined, avg_conf
    
    def _clean_text(self, text: str) -> str:
        """Clean OCR text"""
        # Keep only alphanumeric and common separators
        cleaned = re.sub(r'[^A-Za-z0-9\s\-]', '', text)
        cleaned = ' '.join(cleaned.split())
        return cleaned.upper()
    
    def crop_plate(self, frame: np.ndarray, bbox: List[float], padding: float = 0.1) -> np.ndarray:
        """Crop license plate from frame with padding"""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox]
        
        # Add padding
        pad_w = int((x2 - x1) * padding)
        pad_h = int((y2 - y1) * padding)
        
        x1 = max(0, x1 - pad_w)
        y1 = max(0, y1 - pad_h)
        x2 = min(w, x2 + pad_w)
        y2 = min(h, y2 + pad_h)
        
        return frame[y1:y2, x1:x2].copy()


# Singleton instance
_ocr_instance: Optional[PlateOCR] = None


def get_ocr(use_gpu: bool = True) -> Optional[PlateOCR]:
    """Get or create OCR instance"""
    global _ocr_instance
    
    if not EASYOCR_AVAILABLE:
        return None
    
    if _ocr_instance is None:
        try:
            _ocr_instance = PlateOCR(use_gpu=use_gpu)
        except Exception as e:
            print(f"Failed to initialize OCR: {e}")
            return None
    
    return _ocr_instance


def read_license_plate(frame: np.ndarray, bbox: List[float], use_gpu: bool = True) -> Tuple[str, float]:
    """
    Convenience function to read a license plate.
    
    Args:
        frame: Full BGR frame
        bbox: Plate bounding box [x1, y1, x2, y2]
        
    Returns:
        Tuple of (plate_text, confidence)
    """
    ocr = get_ocr(use_gpu)
    
    if ocr is None:
        return "", 0.0
    
    plate_img = ocr.crop_plate(frame, bbox)
    return ocr.read_plate(plate_img)


# Test
if __name__ == "__main__":
    if EASYOCR_AVAILABLE:
        # Create test image with text
        test_img = np.ones((60, 200, 3), dtype=np.uint8) * 255
        cv2.putText(test_img, 'ABC 1234', (10, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        ocr = PlateOCR(use_gpu=True)
        text, conf = ocr.read_plate(test_img)
        print(f"Test plate: '{text}' (confidence: {conf:.2f})")
    else:
        print("EasyOCR not available. Install with: pip install easyocr")