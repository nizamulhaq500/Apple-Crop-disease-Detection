import torch
from pathlib import Path
from data_loader import get_dataloaders
from model import get_model
from train import get_device

# Default paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

def evaluate_model():
    print("Setting up evaluation...")
    device = get_device()
    print(f"Using device: {device}")
    
    # 1. Load Data
    _, _, test_loader, classes = get_dataloaders()
    num_classes = len(classes)
    
    # 2. Initialize Model
    model = get_model(num_classes)
    
    # 3. Load best weights
    best_model_path = MODELS_DIR / "best_model.pth"
    if not best_model_path.exists():
        raise FileNotFoundError(f"Could not find model weights at {best_model_path}. Did you train the model?")
        
    model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()
    
    # 4. Evaluation Loop
    test_corrects = 0
    total_samples = 0
    
    # Optional: Per-class accuracy
    class_correct = list(0. for _ in range(num_classes))
    class_total = list(0. for _ in range(num_classes))
    
    print("\nStarting evaluation on test set...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            test_corrects += torch.sum(preds == labels.data)
            total_samples += inputs.size(0)
            
            c = (preds == labels.data).squeeze()
            for i in range(len(labels)):
                label = labels[i].item()
                class_correct[label] += c[i].item()
                class_total[label] += 1
                
    test_acc = test_corrects.float() / total_samples
    print("-" * 20)
    print(f"Overall Test Accuracy: {test_acc:.4f} ({int(test_corrects)}/{total_samples})")
    print("-" * 20)
    
    print("\nPer-class Accuracy:")
    for i in range(num_classes):
        if class_total[i] > 0:
            acc = 100 * class_correct[i] / class_total[i]
            print(f"{classes[i]:25s}: {acc:.1f}% ({int(class_correct[i])}/{int(class_total[i])})")
        else:
            print(f"{classes[i]:25s}: N/A (no samples)")

if __name__ == "__main__":
    evaluate_model()
