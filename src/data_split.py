from pathlib import Path
import random
import shutil

# We can define paths relative to this script
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE = PROJECT_ROOT / "data" / "apple_raw"
DEFAULT_SPLIT = PROJECT_ROOT / "data" / "split"

def split_data(source_dir_path=DEFAULT_SOURCE, split_dir_path=DEFAULT_SPLIT, split_ratios=None, seed=42):
    """
    Reads images from source_dir_path, splits them into train/val/test folders,
    and writes them to split_dir_path.
    """
    if split_ratios is None:
        split_ratios = {"train": 0.70, "val": 0.15, "test": 0.15}
    
    source_dir = Path(source_dir_path)
    split_dir = Path(split_dir_path)
    
    random.seed(seed)
    
    print(f"Source directory: {source_dir}")
    print(f"Output directory: {split_dir}")
    
    if split_dir.exists():
        print("Clearing existing split directory...")
        shutil.rmtree(split_dir)
        
    image_extensions = {".jpg", ".jpeg", ".png"}
    
    for class_dir in source_dir.iterdir():
        if not class_dir.is_dir():
            continue
            
        images = [
            path for path in class_dir.iterdir()
            if path.suffix.lower() in image_extensions
        ]
        
        random.shuffle(images)
        total = len(images)
        
        train_end = int(total * split_ratios["train"])
        val_end = train_end + int(total * split_ratios["val"])
        
        split_images = {
            "train": images[:train_end],
            "val": images[train_end:val_end],
            "test": images[val_end:]
        }
        
        for split_name, image_list in split_images.items():
            output_folder = split_dir / split_name / class_dir.name
            output_folder.mkdir(parents=True, exist_ok=True)
            
            for image_path in image_list:
                shutil.copy2(image_path, output_folder / image_path.name)
                
            print(f"{split_name:5} | {class_dir.name:30} | {len(image_list):>4} images")

if __name__ == "__main__":
    split_data()
