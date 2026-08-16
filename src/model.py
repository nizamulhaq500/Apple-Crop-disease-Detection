import torch.nn as nn
from torchvision import models

def get_model(num_classes):
    """
    Returns a pre-trained ResNet18 model with the final layer modified
    to output the specified number of classes.
    """
    # Load a pre-trained ResNet18 model
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    
    # Freeze the early layers (optional, but good for fine-tuning)
    for param in model.parameters():
        param.requires_grad = False
        
    # Replace the final fully connected layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    # Ensure the new layer is trainable
    for param in model.fc.parameters():
        param.requires_grad = True
        
    return model
