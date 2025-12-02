
import cv2
import numpy as np
from typing import Dict, Any, Tuple, List
from datetime import datetime
from pathlib import Path

from infer_yolo import YOLODetector, load_model, Detection
from postprocess import analyze_detections, FrameAnalysis, Motorcycle
from ocr_utils import read_license_plate, EASYOCR_AVAILABLE


# color palette in bgr because opencv flips channels
COLORS = {
    'motorcycle': (255, 165, 0),     # kind of orange
    'rider': (0, 255, 0),            # bright green
    'helmet': (0, 255, 255),         # banana yellow
    'no_helmet': (0, 0, 255),        # loud red
    'license_plate': (255, 0, 255),  # magenta vibe
    'violation': (0, 0, 255),        # same red for violations
}


def draw_detection(frame: np.ndarray, det: Detection, color: tuple, label: str = None):
    """Draw a single detection on frame"""
    x1, y1, x2, y2 = [int(v) for v in det.bbox]
    
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    
    if label:
        # quick background box so the label is readable
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - text_h - 4), (x1 + text_w, y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 4), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def draw_annotations(frame: np.ndarray, analysis: FrameAnalysis) -> np.ndarray:
    annotated = frame.copy()
    
    for moto in analysis.motorcycles:
        # outline the bike
        moto_color = COLORS['violation'] if moto.has_violation else COLORS['motorcycle']
        thickness = 3 if moto.has_violation else 2
        
        x1, y1, x2, y2 = [int(v) for v in moto.detection.bbox]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), moto_color, thickness)
        
        # label text for the bike
        label = "MOTORCYCLE"
        if moto.has_violation:
            label += f" [{', '.join(moto.violation_types).upper()}]"
        
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(annotated, (x1, y1 - text_h - 8), (x1 + text_w + 4, y1), moto_color, -1)
        cv2.putText(annotated, label, (x1 + 2, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # tag riders and helmets
        for rider in moto.riders:
            r_color = COLORS['helmet'] if rider.has_helmet else COLORS['no_helmet']
            r_label = "HELMET" if rider.has_helmet else "NO HELMET"
            draw_detection(annotated, rider.detection, r_color, r_label)
        
        # throw on plate text if we have it
        if moto.license_plate:
            p_label = moto.plate_text if moto.plate_text else "PLATE"
            draw_detection(annotated, moto.license_plate, COLORS['license_plate'], p_label)
    
    return annotated


def draw_stats_overlay(frame: np.ndarray, stats: Dict[str, Any]) -> np.ndarray:
    h, w = frame.shape[:2]
    
    # slap a translucent box for stats
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (250, 160), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
    
    # stuff we want to show in the overlay
    lines = [
        f"Motorcycles: {stats.get('motorcycles', 0)}",
        f"Total Riders: {stats.get('total_riders', 0)}",
        f"Helmeted: {stats.get('helmeted_riders', 0)}",
        f"No Helmet: {stats.get('unhelmeted_riders', 0)}",
        f"Overloaded: {stats.get('overloaded_motorcycles', 0)}",
        f"Compliance: {stats.get('compliance_rate', 100):.1f}%",
    ]
    
    y = 35
    for line in lines:
        color = (0, 255, 0) if 'Compliance' in line else (255, 255, 255)
        cv2.putText(frame, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        y += 22
    
    return frame


def analyze_frame(
    frame: np.ndarray,
    model: YOLODetector,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    read_plates: bool = True
) -> Tuple[np.ndarray, Dict[str, Any]]:

    # run yolo
    detections = model.predict(frame, conf_threshold, iou_threshold)
    
    # let postprocess sort everyone out
    analysis = analyze_detections(detections)
    
    # try to read plates when there's a violation
    plate_readings = []
    if read_plates and EASYOCR_AVAILABLE:
        for moto in analysis.motorcycles:
            if moto.has_violation and moto.license_plate:
                plate_text, plate_conf = read_license_plate(frame, moto.license_plate.bbox)
                moto.plate_text = plate_text
                
                if plate_text:
                    plate_readings.append({
                        'text': plate_text,
                        'confidence': plate_conf,
                        'violation_types': moto.violation_types,
                        'timestamp': datetime.now().isoformat()
                    })
    
    # doodle annotations back onto the frame
    annotated = draw_annotations(frame, analysis)
    
    # quick compliance math
    compliance_rate = (
        analysis.helmeted_riders / analysis.total_riders * 100 
        if analysis.total_riders > 0 else 100.0
    )
    
    # stash stats for whoever called us
    stats = {
        'motorcycles': len(analysis.motorcycles),
        'total_riders': analysis.total_riders,
        'helmeted_riders': analysis.helmeted_riders,
        'unhelmeted_riders': analysis.unhelmeted_riders,
        'overloaded_motorcycles': analysis.overloaded_count,
        'compliance_rate': compliance_rate,
        'violations': analysis.violations,
        'plate_readings': plate_readings,
        'timestamp': datetime.now().isoformat()
    }
    
    # layer the numbers on top
    annotated = draw_stats_overlay(annotated, stats)
    
    return annotated, stats


def process_image(image_path: str, model: YOLODetector, output_path: str = None) -> Dict[str, Any]:
    """Process a single image file"""
    frame = cv2.imread(image_path)
    if frame is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    annotated, stats = analyze_frame(frame, model)
    
    if output_path:
        cv2.imwrite(output_path, annotated)
        print(f"saved to: {output_path}")
    
    return stats


def process_video(video_path: str, model: YOLODetector, output_path: str = None,
                  frame_skip: int = 1, show_preview: bool = False) -> List[Dict[str, Any]]:
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # grab basic video info
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # spin up a writer if we actually want a file
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps // frame_skip, (width, height))
    
    all_stats = []
    frame_idx = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % frame_skip == 0:
            annotated, stats = analyze_frame(frame, model, read_plates=True)
            all_stats.append(stats)
            
            if writer:
                writer.write(annotated)
            
            if show_preview:
                cv2.imshow('ThinkAHead', annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        frame_idx += 1
    
    cap.release()
    if writer:
        writer.release()
    if show_preview:
        cv2.destroyAllWindows()
    return all_stats


# little test hook
if __name__ == "__main__":
    import sys
    
    model = load_model('models/trained/thinkahead_best.pt')
    
    # quick sanity run on a sample image
    test_dir = Path('data/processed/images/test')
    test_images = list(test_dir.glob('*.jpg')) + list(test_dir.glob('*.png'))
    
    if test_images:
        test_img = str(test_images[0])
        frame = cv2.imread(test_img)
        annotated, stats = analyze_frame(frame, model)
        
        summary = (
            f"sample: {Path(test_img).name} | "
            f"riders: {stats['total_riders']} | "
            f"helmeted: {stats['helmeted_riders']} | "
            f"no helmet: {stats['unhelmeted_riders']} | "
            f"compliance: {stats['compliance_rate']:.1f}%"
        )
        print(summary)
        
        # save result
        output_path = 'outputs/test_result.jpg'
        Path('outputs').mkdir(exist_ok=True)
        cv2.imwrite(output_path, annotated)
        print(f"saved result to: {output_path}")
    else:
        print("no test images found in data/processed/images/test/")
