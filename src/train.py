import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from data_loader import get_dataloaders
from model import get_model

# Default paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def train_model(num_epochs=10, batch_size=32, learning_rate=1e-3):
    print("Setting up training...")
    
    # 1. Ensure models directory exists
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # 2. Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # 3. Load Data
    train_loader, val_loader, _, classes = get_dataloaders(batch_size=batch_size)
    num_classes = len(classes)
    
    # 4. Initialize Model, Loss, and Optimizer
    model = get_model(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    # We only optimize the parameters that require gradients (the new fc layer)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=learning_rate
    )
    
    # 5. Training Loop
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 10)
        
        # --- Training Phase ---
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.float() / len(train_loader.dataset)
        print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
        
        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)
                
        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_acc = val_corrects.float() / len(val_loader.dataset)
        print(f"Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f}")
        
        # Save best model
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            best_model_path = MODELS_DIR / "best_model.pth"
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved new best model with Val Acc: {best_val_acc:.4f}")

    print("\nTraining complete!")
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")

if __name__ == "__main__":
    # For a quick verification, we can run with num_epochs=1
    train_model(num_epochs=1)
