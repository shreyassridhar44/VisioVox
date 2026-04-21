# models/visual_model.py
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
import logging

# Basic logger for standalone testing
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("VisualModel")

class VisualEncoder(nn.Module):
    def __init__(self, embedding_dim=128):
        super(VisualEncoder, self).__init__()
        
        logger.info("Initializing Pretrained ResNet-18...")
        # Load the pretrained ResNet-18 model
        resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        
        # 1. Modify the first convolutional layer
        # Standard ResNet takes 3 channels (RGB). We change it to 1 channel (Grayscale).
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Pro-tip: Copy the averaged weights from the original RGB layer so we don't start from scratch
        with torch.no_grad():
            self.conv1.weight[:] = resnet.conv1.weight.mean(dim=1, keepdim=True)
            
        # 2. Copy the rest of the standard ResNet layers
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
        
        # 3. Replace the final classification head
        # We don't want it to classify 1000 ImageNet classes; we want a custom embedding vector.
        self.fc = nn.Linear(resnet.fc.in_features, embedding_dim)
        
    def forward(self, x):
        # Input shape expected: (Batch Size, Channels=1, Height=112, Width=112)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # Flatten spatial dimensions
        x = self.fc(x)           # Output the embedding
        
        return x

# --- Standalone Test ---
if __name__ == "__main__":
    # Let's run a dummy test to ensure PyTorch compiles it and tensor math works
    try:
        model = VisualEncoder(embedding_dim=128)
        logger.info("Model initialized successfully.")
        
        # Create a dummy tensor representing a batch of 2 grayscale lip frames (112x112)
        # Shape: (Batch_Size=2, Channels=1, Height=112, Width=112)
        dummy_input = torch.randn(2, 1, 112, 112)
        logger.info(f"Passing dummy input with shape: {dummy_input.shape}")
        
        # Pass it through the model
        output = model(dummy_input)
        
        logger.info(f"Forward pass successful! Output shape: {output.shape}")
        logger.info("Expected shape is [2, 128]. If it matches, we are good to go!")
        
    except Exception as e:
        logger.error(f"Error during model testing: {e}")