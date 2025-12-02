from ultralytics import YOLO
from pathlib import Path
import argparse
import torch
import yaml
import sys
import gc


def check_gpu():
    
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {gpu_name}")
        print(f"GPU Memory: {gpu_mem:.1f} GB")
        
        if gpu_mem < 4:
            return 4, 0
        elif gpu_mem < 6:
            return 8, 0
        else:
            return 16, 0



def find_data_yaml():
    """Find the processed data.yaml"""
    candidates = [
        Path('data/processed/data.yaml'),
        Path('data/data.yaml'),
    ]
    
    for p in candidates:
        if p.exists():
            return p
    
    return None


def validate_dataset(yaml_path: Path):
    
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    
    base_path = Path(config.get('path', yaml_path.parent))
    
    print(f"  Classes ({config.get('nc', '?')}):")
    names = config.get('names', [])
    for i, name in enumerate(names):
        print(f"    {i}: {name}")
    
    # Check splits
    splits_ok = True
    for split in ['train', 'val', 'test']:
        split_path = config.get(split, f'images/{split}')
        full_path = base_path / split_path
        
        if full_path.exists():
            count = len(list(full_path.glob('*')))
            print(f"  {split}: {count} images")
        else:
            print(f"  {split}: NOT FOUND at {full_path}")
            if split == 'train':
                splits_ok = False
    
    return splits_ok


def train(data_yaml, epochs=100, batch_size=8, img_size=640, 
          model_size='m', device=0, resume=False, patience=30):
    
    # Clear memory before starting
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print("\n" + "=" * 50)
    print("Training Configuration (Memory Optimized)")
    print("=" * 50)
    
    model_name = f'yolov8{model_size}.pt'
    print(f"Model: {model_name}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Image size: {img_size}")
    
    model = YOLO(model_name)
    
    output_dir = Path('outputs/runs')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nStarting training...\n")
    
    results = model.train(
        data=str(Path(data_yaml).absolute()),
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=device,
        
        # Optimizer
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.01,
        warmup_epochs=3,
        
        # Augmentation
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,  # Disabled to save memory
        
        # Training
        patience=patience,
        save=True,
        save_period=10,
        

        cache=False, 
        workers=2,
        
        # Output
        project=str(output_dir),
        name='thinkahead',
        exist_ok=True,
        
        # Misc
        verbose=True,
        seed=42,
        resume=resume,
    )
    
    # Save best model
    best_path = output_dir / 'thinkahead' / 'weights' / 'best.pt'
    if best_path.exists():
        models_dir = Path('models/trained')
        models_dir.mkdir(parents=True, exist_ok=True)
        import shutil
        dest = models_dir / 'thinkahead_best.pt'
        shutil.copy(best_path, dest)
        print(f"\n✓ Best model saved to: {dest}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Train ThinkAHead YOLOv8')
    parser.add_argument('--data', type=str, help='Path to data.yaml')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch', type=int, default=None)
    parser.add_argument('--img-size', type=int, default=640)
    parser.add_argument('--model', type=str, default='m', choices=['n','s','m','l','x'])
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--test', action='store_true', help='Quick 5-epoch test')
    parser.add_argument('--patience', type=int, default=30)
    
    args = parser.parse_args()
    
    # Check GPU
    rec_batch, device = check_gpu()
    
    # Find data.yaml
    if args.data:
        data_yaml = Path(args.data)
    else:
        data_yaml = find_data_yaml()
    
    
    epochs = 5 if args.test else args.epochs
    batch = args.batch if args.batch else rec_batch
    
    if args.test:
        print("\n🧪 TEST MODE: 5 epochs only")
    
    
    try:
        train(
            data_yaml=data_yaml,
            epochs=epochs,
            batch_size=batch,
            img_size=args.img_size,
            model_size=args.model,
            device=device,
            resume=args.resume,
            patience=args.patience
        )


if __name__ == "__main__":
    main()