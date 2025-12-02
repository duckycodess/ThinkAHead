#!/usr/bin/env python3

from typing import List, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class Rider:
    """Represents a rider on a motorcycle"""
    detection: object  # Detection object
    has_helmet: bool = False
    helmet_detection: object = None


@dataclass 
class Motorcycle:
    """Represents a motorcycle with its riders and plate"""
    detection: object
    riders: List[Rider] = field(default_factory=list)
    license_plate: object = None
    plate_text: str = ""
    is_overloaded: bool = False
    has_violation: bool = False
    violation_types: List[str] = field(default_factory=list)


@dataclass
class FrameAnalysis:
    """Complete analysis of a single frame"""
    motorcycles: List[Motorcycle]
    total_riders: int
    helmeted_riders: int
    unhelmeted_riders: int
    overloaded_count: int
    violations: List[Dict]


def boxes_overlap(det1, det2, threshold: float = 0.3) -> bool:
    """Check if two detections overlap significantly"""
    x1 = max(det1.x1, det2.x1)
    y1 = max(det1.y1, det2.y1)
    x2 = min(det1.x2, det2.x2)
    y2 = min(det1.y2, det2.y2)
    
    if x2 <= x1 or y2 <= y1:
        return False
    
    intersection = (x2 - x1) * (y2 - y1)
    smaller_area = min(det1.area, det2.area)
    
    return (intersection / smaller_area) >= threshold if smaller_area > 0 else False


def is_inside(inner, outer, threshold: float = 0.5) -> bool:
    """Check if inner detection is mostly inside outer detection"""
    x1 = max(inner.x1, outer.x1)
    y1 = max(inner.y1, outer.y1)
    x2 = min(inner.x2, outer.x2)
    y2 = min(inner.y2, outer.y2)
    
    if x2 <= x1 or y2 <= y1:
        return False
    
    intersection = (x2 - x1) * (y2 - y1)
    return (intersection / inner.area) >= threshold if inner.area > 0 else False


def is_helmet_on_rider(helmet, rider) -> bool:
    """Check if a helmet detection is on the rider's head (upper portion)"""
    # Helmet should be in upper 40% of rider bbox
    rider_head_y = rider.y1 + rider.height * 0.4
    
    if helmet.center_y > rider_head_y:
        return False
    
    # Check horizontal overlap
    x_overlap_start = max(helmet.x1, rider.x1)
    x_overlap_end = min(helmet.x2, rider.x2)
    
    if x_overlap_end <= x_overlap_start:
        return False
    
    x_overlap = x_overlap_end - x_overlap_start
    overlap_ratio = x_overlap / helmet.width if helmet.width > 0 else 0
    
    return overlap_ratio >= 0.3


def is_rider_on_motorcycle(rider, motorcycle) -> bool:
    """Check if a rider is associated with a motorcycle"""
    # Check direct overlap
    if boxes_overlap(rider, motorcycle, threshold=0.2):
        return True
    
    # Check if rider is above/on motorcycle
    if is_inside(rider, motorcycle, threshold=0.3):
        return True
    
    # Check vertical alignment and proximity
    horizontal_overlap = not (rider.x2 < motorcycle.x1 or rider.x1 > motorcycle.x2)
    
    if horizontal_overlap:
        # Rider's bottom should be near motorcycle's vertical range
        vertical_distance = abs(rider.y2 - motorcycle.center_y)
        if vertical_distance < motorcycle.height * 0.8:
            return True
    
    return False


def is_plate_on_motorcycle(plate, motorcycle) -> bool:
    """Check if a license plate belongs to a motorcycle"""
    # Plate should overlap or be very close to motorcycle
    if boxes_overlap(plate, motorcycle, threshold=0.2):
        return True
    
    if is_inside(plate, motorcycle, threshold=0.5):
        return True
    
    # Plate often at bottom of motorcycle
    if plate.center_y > motorcycle.center_y:
        horizontal_dist = abs(plate.center_x - motorcycle.center_x)
        if horizontal_dist < motorcycle.width * 0.6:
            vertical_dist = plate.y1 - motorcycle.y2
            if vertical_dist < motorcycle.height * 0.4:
                return True
    
    return False


def analyze_detections(detections: List) -> FrameAnalysis:
    """
    Analyze detections to identify motorcycles, riders, helmets, and violations.
    
    Args:
        detections: List of Detection objects from YOLO inference
        
    Returns:
        FrameAnalysis with complete breakdown
    """
    # Separate by class
    motorcycles_det = [d for d in detections if d.class_name == 'motorcycle']
    riders_det = [d for d in detections if d.class_name == 'rider']
    helmets_det = [d for d in detections if d.class_name == 'helmet']
    no_helmets_det = [d for d in detections if d.class_name == 'no_helmet']
    plates_det = [d for d in detections if d.class_name == 'license_plate']
    
    # Combine rider and no_helmet as "people" (no_helmet IS a rider without helmet)
    # helmet class means person WITH helmet
    all_people = riders_det + no_helmets_det + helmets_det
    
    motorcycles = []
    assigned_people = set()
    assigned_plates = set()
    
    # Process each motorcycle
    for moto_det in motorcycles_det:
        motorcycle = Motorcycle(detection=moto_det)
        
        # Find riders on this motorcycle
        # Check riders_det (generic rider)
        for i, person_det in enumerate(all_people):
            person_id = id(person_det)
            if person_id in assigned_people:
                continue
            
            if is_rider_on_motorcycle(person_det, moto_det):
                # Determine if this person has helmet based on class
                if person_det.class_name == 'helmet':
                    # This detection IS a helmeted person
                    rider = Rider(detection=person_det, has_helmet=True, helmet_detection=person_det)
                elif person_det.class_name == 'no_helmet':
                    # This detection IS a non-helmeted person
                    rider = Rider(detection=person_det, has_helmet=False)
                else:
                    # Generic rider - check for nearby helmet
                    rider = Rider(detection=person_det, has_helmet=False)
                    for h_det in helmets_det:
                        if is_helmet_on_rider(h_det, person_det):
                            rider.has_helmet = True
                            rider.helmet_detection = h_det
                            break
                
                motorcycle.riders.append(rider)
                assigned_people.add(person_id)
        
        # Find license plate for this motorcycle
        for plate_det in plates_det:
            plate_id = id(plate_det)
            if plate_id in assigned_plates:
                continue
            
            if is_plate_on_motorcycle(plate_det, moto_det):
                motorcycle.license_plate = plate_det
                assigned_plates.add(plate_id)
                break
        
        # Determine violations
        motorcycle.is_overloaded = len(motorcycle.riders) >= 3
        unhelmeted = [r for r in motorcycle.riders if not r.has_helmet]
        
        if motorcycle.is_overloaded:
            motorcycle.has_violation = True
            motorcycle.violation_types.append('overloaded')
        
        if unhelmeted:
            motorcycle.has_violation = True
            motorcycle.violation_types.append('no_helmet')
        
        motorcycles.append(motorcycle)
    
    # Handle unassigned people (not on any motorcycle)
    # These might be standalone detections - still count them
    standalone_helmeted = 0
    standalone_unhelmeted = 0
    
    for person_det in all_people:
        if id(person_det) not in assigned_people:
            if person_det.class_name == 'helmet':
                standalone_helmeted += 1
            elif person_det.class_name == 'no_helmet':
                standalone_unhelmeted += 1
    
    # Compile statistics
    total_on_motorcycles = sum(len(m.riders) for m in motorcycles)
    helmeted_on_motorcycles = sum(1 for m in motorcycles for r in m.riders if r.has_helmet)
    
    total_riders = total_on_motorcycles + standalone_helmeted + standalone_unhelmeted
    helmeted_riders = helmeted_on_motorcycles + standalone_helmeted
    unhelmeted_riders = total_riders - helmeted_riders
    overloaded = sum(1 for m in motorcycles if m.is_overloaded)
    
    # Compile violations
    violations = []
    for moto in motorcycles:
        if moto.has_violation:
            violations.append({
                'motorcycle_bbox': moto.detection.bbox,
                'violation_types': moto.violation_types,
                'rider_count': len(moto.riders),
                'unhelmeted_count': len([r for r in moto.riders if not r.has_helmet]),
                'plate_bbox': moto.license_plate.bbox if moto.license_plate else None,
                'plate_confidence': moto.license_plate.confidence if moto.license_plate else None
            })
    
    return FrameAnalysis(
        motorcycles=motorcycles,
        total_riders=total_riders,
        helmeted_riders=helmeted_riders,
        unhelmeted_riders=unhelmeted_riders,
        overloaded_count=overloaded,
        violations=violations
    )


# Test
if __name__ == "__main__":
    from infer_yolo import Detection
    
    # Create mock detections
    mock_detections = [
        Detection([100, 200, 400, 500], 0, 0.9),   # motorcycle
        Detection([150, 150, 300, 400], 1, 0.85),  # rider
        Detection([170, 140, 250, 200], 2, 0.8),   # helmet
        Detection([200, 450, 300, 490], 4, 0.75),  # license_plate
    ]
    
    analysis = analyze_detections(mock_detections)
    
    print(f"Motorcycles: {len(analysis.motorcycles)}")
    print(f"Total riders: {analysis.total_riders}")
    print(f"Helmeted: {analysis.helmeted_riders}")
    print(f"Unhelmeted: {analysis.unhelmeted_riders}")
    print(f"Overloaded: {analysis.overloaded_count}")
    print(f"Violations: {len(analysis.violations)}")