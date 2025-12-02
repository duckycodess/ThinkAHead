

import os
import shutil
import yaml
import random
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict
from PIL import Image
from tqdm import tqdm

# ============================================================================
# UNIFIED CLASS MAPPING FOR THINKAHEAD
# ============================================================================
# We'll use 5 classes that cover all detection needs:
#   0: motorcycle   - The bike itself
#   1: rider        - Person on motorcycle (helmet status unknown)
#   2: helmet       - Person WITH helmet OR helmet object
#   3: no_helmet    - Person WITHOUT helmet
#   4: license_plate - The plate

UNIFIED_CLASSES = ['motorcycle', 'rider', 'helmet', 'no_helmet', 'license_plate']

# Mapping from various dataset class names to our unified IDs
# Format: 'original_class_name': unified_class_id
CLASS_MAPPINGS = {
    # From Helmet-and-Number-plate dataset
    'helmet': 2,
    'non-helmet': 3,
    'non_helmet': 3,
    'non_rider': 1,      # Still a person, treat as rider
    'rider': 1,
    'licence_plate': 4,
    'license_plate': 4,
    'license plate': 4,
    'number_plate': 4,
    'numberplate': 4,
    'plate': 4,
    
    # From Kaggle helmet detection
    'with helmet': 2,
    'with_helmet': 2,
    'without helmet': 3,
    'without_helmet': 3,
    'head': 3,           # Bare head = no helmet
    
    # From triple riding / violations datasets
    'motorcycle': 0,
    'motorbike': 0,
    'bike': 0,
    'triple': 0,         # Triple riding = still a motorcycle
    'triple-ride': 0,
    'triple_ride': 0,
    'triple riding': 0,
    'triple-riding': 0,
    'triples': 0,
    'overloaded': 0,
    
    # Generic person classes
    'person': 1,
    'motorcyclist': 1,
    
    # Violations dataset specific
    'with helmet': 2,
    'without helmet': 3,
    'no plate': 4,       # Map to plate class (will detect absence)
    'phone usage': 1,    # Treat as rider
    'stunt riding': 0,   # Treat as motorcycle
}


def find_datasets(project_root: Path) -> dict:
    """Find all available datasets in the project"""
    datasets = {}
    
    # Possible locations to check
    locations = [
        # Roboflow datasets in data/raw
        ('roboflow_violations', project_root / 'data' / 'raw' / 'roboflow_violations'),
        ('roboflow_helmet_plate', project_root / 'data' / 'raw' / 'roboflow_helmet_plate'),
        ('roboflow_triple_riding', project_root / 'data' / 'raw' / 'roboflow_triple_riding'),
        
        # Helmet-and-Number-plate (might be in root or data/raw)
        ('helmet_plate_v1', project_root / 'Helmet-and-Number-plate--1'),
        ('helmet_plate_v2', project_root / 'data' / 'raw' / 'Helmet-and-Number-plate--1'),
        ('helmet_plate_v3', project_root / 'data' / 'raw' / 'helmet_plate'),
        
        # Kaggle dataset
        ('kaggle_helmet', project_root / 'data' / 'raw' / 'kaggle_helmet'),
    ]
    
    # Also check for annotations/images at root (Kaggle format)
    kaggle_annotations = project_root / 'annotations'
    kaggle_images = project_root / 'images'
    if kaggle_annotations.exists() and kaggle_images.exists():
        datasets['kaggle_root'] = {
            'path': project_root,
            'type': 'kaggle_voc',
            'annotations': kaggle_annotations,
            'images': kaggle_images
        }
    
    for name, path in locations:
        if path.exists():
            # Determine dataset type
            if (path / 'train' / 'images').exists() or (path / 'train' / 'labels').exists():
                datasets[name] = {'path': path, 'type': 'roboflow_yolo'}
            elif (path / 'images').exists() and (path / 'labels').exists():
                datasets[name] = {'path': path, 'type': 'yolo_flat'}
            elif (path / 'annotations').exists():
                datasets[name] = {'path': path, 'type': 'kaggle_voc'}
    
    return datasets


def get_class_mapping_from_yaml(yaml_path: Path) -> dict:
    """Read class names from a data.yaml file"""
    if not yaml_path.exists():
        return {}
    
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    
    names = config.get('names', [])
    
    # Handle both list and dict formats
    if isinstance(names, list):
        return {i: name.lower().strip() for i, name in enumerate(names)}
    elif isinstance(names, dict):
        return {int(k): v.lower().strip() for k, v in names.items()}
    
    return {}


def remap_class_id(original_id: int, original_classes: dict) -> int:
    """Remap a class ID from original dataset to unified scheme"""
    if original_id not in original_classes:
        return -1  # Unknown class
    
    original_name = original_classes[original_id].lower().strip()
    
    if original_name in CLASS_MAPPINGS:
        return CLASS_MAPPINGS[original_name]
    
    # Try partial matching
    for key, value in CLASS_MAPPINGS.items():
        if key in original_name or original_name in key:
            return value
    
    print(f"  ⚠ Unknown class: '{original_name}' (id={original_id})")
    return -1


def convert_voc_to_yolo(xml_path: Path, img_width: int, img_height: int) -> list:
    """Convert Pascal VOC XML to YOLO format lines"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    lines = []
    
    for obj in root.findall('object'):
        class_name = obj.find('name').text.lower().strip()
        
        if class_name not in CLASS_MAPPINGS:
            print(f"  ⚠ Unknown VOC class: '{class_name}'")
            continue
        
        class_id = CLASS_MAPPINGS[class_name]
        
        bbox = obj.find('bndbox')
        xmin = float(bbox.find('xmin').text)
        ymin = float(bbox.find('ymin').text)
        xmax = float(bbox.find('xmax').text)
        ymax = float(bbox.find('ymax').text)
        
        # Convert to YOLO format (normalized center x, y, width, height)
        x_center = (xmin + xmax) / 2 / img_width
        y_center = (ymin + ymax) / 2 / img_height
        width = (xmax - xmin) / img_width
        height = (ymax - ymin) / img_height
        
        # Clamp to [0, 1]
        x_center = max(0, min(1, x_center))
        y_center = max(0, min(1, y_center))
        width = max(0, min(1, width))
        height = max(0, min(1, height))
        
        lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    
    return lines


def process_kaggle_dataset(dataset_info: dict, output_dir: Path, prefix: str) -> list:
    """Process Kaggle VOC format dataset"""
    print(f"\n📂 Processing Kaggle dataset: {prefix}")
    
    if 'annotations' in dataset_info:
        annotations_dir = dataset_info['annotations']
        images_dir = dataset_info['images']
    else:
        annotations_dir = dataset_info['path'] / 'annotations'
        images_dir = dataset_info['path'] / 'images'
    
    if not annotations_dir.exists():
        print(f"  ❌ Annotations not found: {annotations_dir}")
        return []
    
    processed = []
    xml_files = list(annotations_dir.glob('*.xml'))
    
    print(f"  Found {len(xml_files)} annotation files")
    
    for xml_path in tqdm(xml_files, desc=f"  Converting {prefix}"):
        # Find corresponding image
        img_name = xml_path.stem
        img_path = None
        
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            candidate = images_dir / f"{img_name}{ext}"
            if candidate.exists():
                img_path = candidate
                break
        
        if img_path is None:
            continue
        
        # Get image dimensions
        try:
            with Image.open(img_path) as img:
                img_width, img_height = img.size
        except Exception as e:
            print(f"  ⚠ Could not read image: {img_path}")
            continue
        
        # Convert annotations
        yolo_lines = convert_voc_to_yolo(xml_path, img_width, img_height)
        
        if not yolo_lines:
            continue
        
        # Copy image
        new_img_name = f"{prefix}_{img_path.name}"
        shutil.copy(img_path, output_dir / 'images' / new_img_name)
        
        # Write label
        label_name = f"{prefix}_{img_path.stem}.txt"
        with open(output_dir / 'labels' / label_name, 'w') as f:
            f.write('\n'.join(yolo_lines))
        
        processed.append(new_img_name)
    
    print(f"  ✓ Processed {len(processed)} images")
    return processed


def process_roboflow_dataset(dataset_info: dict, output_dir: Path, prefix: str) -> list:
    """Process Roboflow YOLO format dataset"""
    print(f"\n📂 Processing Roboflow dataset: {prefix}")
    
    dataset_path = dataset_info['path']
    
    # Read class mapping from data.yaml
    yaml_path = dataset_path / 'data.yaml'
    original_classes = get_class_mapping_from_yaml(yaml_path)
    
    if original_classes:
        print(f"  Original classes: {original_classes}")
    
    processed = []
    
    # Process each split (train, valid, test)
    for split in ['train', 'valid', 'test']:
        split_images = dataset_path / split / 'images'
        split_labels = dataset_path / split / 'labels'
        
        if not split_images.exists():
            continue
        
        image_files = list(split_images.glob('*.[jJpP][pPnN][gG]')) + \
                      list(split_images.glob('*.jpeg')) + \
                      list(split_images.glob('*.JPEG'))
        
        for img_path in tqdm(image_files, desc=f"  {prefix}/{split}"):
            label_path = split_labels / f"{img_path.stem}.txt"
            
            if not label_path.exists():
                # Copy image without labels (background image)
                new_img_name = f"{prefix}_{split}_{img_path.name}"
                shutil.copy(img_path, output_dir / 'images' / new_img_name)
                # Create empty label file
                with open(output_dir / 'labels' / f"{prefix}_{split}_{img_path.stem}.txt", 'w') as f:
                    pass
                processed.append(new_img_name)
                continue
            
            # Read and remap labels
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            new_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                
                original_id = int(parts[0])
                
                if original_classes:
                    new_id = remap_class_id(original_id, original_classes)
                else:
                    # If no yaml, assume classes are already in our format
                    new_id = original_id if original_id < len(UNIFIED_CLASSES) else -1
                
                if new_id >= 0:
                    new_lines.append(f"{new_id} {' '.join(parts[1:])}")
            
            if not new_lines:
                continue
            
            # Copy image
            new_img_name = f"{prefix}_{split}_{img_path.name}"
            shutil.copy(img_path, output_dir / 'images' / new_img_name)
            
            # Write remapped labels
            label_name = f"{prefix}_{split}_{img_path.stem}.txt"
            with open(output_dir / 'labels' / label_name, 'w') as f:
                f.write('\n'.join(new_lines))
            
            processed.append(new_img_name)
    
    print(f"  ✓ Processed {len(processed)} images")
    return processed


def create_splits(all_images: list, output_base: Path, 
                  train_ratio: float = 0.8, val_ratio: float = 0.1) -> dict:
    """Split images into train/val/test sets"""
    print("\n📊 Creating train/val/test splits...")
    
    random.seed(42)  # For reproducibility
    random.shuffle(all_images)
    
    n = len(all_images)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    splits = {
        'train': all_images[:n_train],
        'val': all_images[n_train:n_train + n_val],
        'test': all_images[n_train + n_val:]
    }
    
    # Create split directories
    for split_name in ['train', 'val', 'test']:
        (output_base / 'images' / split_name).mkdir(parents=True, exist_ok=True)
        (output_base / 'labels' / split_name).mkdir(parents=True, exist_ok=True)
    
    # Move files to splits
    temp_images = output_base / 'images'
    temp_labels = output_base / 'labels'
    
    for split_name, images in splits.items():
        for img_name in tqdm(images, desc=f"  Moving to {split_name}"):
            # Move image
            src_img = temp_images / img_name
            if src_img.exists():
                shutil.move(str(src_img), str(temp_images / split_name / img_name))
            
            # Move label
            label_name = Path(img_name).stem + '.txt'
            src_label = temp_labels / label_name
            if src_label.exists():
                shutil.move(str(src_label), str(temp_labels / split_name / label_name))
    
    # Clean up any remaining files in temp directories
    for f in temp_images.glob('*.*'):
        if f.is_file():
            f.unlink()
    for f in temp_labels.glob('*.txt'):
        if f.is_file():
            f.unlink()
    
    print(f"  Train: {len(splits['train'])}")
    print(f"  Val:   {len(splits['val'])}")
    print(f"  Test:  {len(splits['test'])}")
    
    return splits


def create_data_yaml(output_base: Path) -> Path:
    """Create the final data.yaml configuration"""
    config = {
        'path': str(output_base.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': len(UNIFIED_CLASSES),
        'names': UNIFIED_CLASSES
    }
    
    yaml_path = output_base / 'data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"\n✓ Created {yaml_path}")
    return yaml_path


def count_class_distribution(output_base: Path) -> dict:
    """Count instances of each class in the dataset"""
    counts = defaultdict(int)
    
    for split in ['train', 'val', 'test']:
        labels_dir = output_base / 'labels' / split
        if not labels_dir.exists():
            continue
        
        for label_file in labels_dir.glob('*.txt'):
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        class_id = int(parts[0])
                        if class_id < len(UNIFIED_CLASSES):
                            counts[UNIFIED_CLASSES[class_id]] += 1
    
    return dict(counts)


def main():
    print("=" * 60)
    print("ThinkAHead Dataset Preparation")
    print("=" * 60)
    
    # Determine project root
    project_root = Path.cwd()
    print(f"\nProject root: {project_root}")
    
    # Create output directory
    output_base = project_root / 'data' / 'processed'
    if output_base.exists():
        print(f"\n⚠ Output directory exists: {output_base}")
        response = input("  Delete and recreate? (y/n): ").strip().lower()
        if response == 'y':
            shutil.rmtree(output_base)
        else:
            print("  Aborting.")
            return
    
    output_base.mkdir(parents=True, exist_ok=True)
    (output_base / 'images').mkdir(exist_ok=True)
    (output_base / 'labels').mkdir(exist_ok=True)
    
    # Find all datasets
    print("\n🔍 Searching for datasets...")
    datasets = find_datasets(project_root)
    
    if not datasets:
        print("❌ No datasets found!")
        print("\nExpected locations:")
        print("  - data/raw/roboflow_violations/")
        print("  - data/raw/roboflow_helmet_plate/")
        print("  - data/raw/roboflow_triple_riding/")
        print("  - Helmet-and-Number-plate--1/")
        print("  - annotations/ + images/ (Kaggle)")
        return
    
    print(f"\nFound {len(datasets)} datasets:")
    for name, info in datasets.items():
        print(f"  ✓ {name}: {info['path']} ({info['type']})")
    
    # Process each dataset
    all_images = []
    
    for name, info in datasets.items():
        if info['type'] == 'kaggle_voc':
            images = process_kaggle_dataset(info, output_base, name)
        elif info['type'] in ['roboflow_yolo', 'yolo_flat']:
            images = process_roboflow_dataset(info, output_base, name)
        else:
            print(f"  ⚠ Unknown type: {info['type']}")
            continue
        
        all_images.extend(images)
    
    print(f"\n📊 Total images collected: {len(all_images)}")
    
    if len(all_images) == 0:
        print("❌ No images were processed!")
        return
    
    # Create train/val/test splits
    splits = create_splits(all_images, output_base)
    
    # Create data.yaml
    yaml_path = create_data_yaml(output_base)
    
    # Count class distribution
    print("\n📈 Class distribution:")
    counts = count_class_distribution(output_base)
    for class_name, count in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {class_name}: {count}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("✓ Dataset preparation complete!")
    print("=" * 60)
    print(f"\nOutput directory: {output_base}")
    print(f"Data config: {yaml_path}")
    print(f"\nTo train, run:")
    print(f"  python train_simple.py --data {yaml_path}")
    
    # Print data.yaml contents
    print(f"\ndata.yaml contents:")
    print("-" * 40)
    with open(yaml_path, 'r') as f:
        print(f.read())


if __name__ == "__main__":
    main()