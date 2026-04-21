# training/train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path

# Import our custom modules
from models.fusion_model import VisioVox
from training.dataset_loader import VisioVoxDataset
from utils.logger import get_logger

logger = get_logger("Trainer")

def train_model(metadata_path: str, epochs: int = 20, batch_size: int = 4, learning_rate: float = 1e-4, save_dir: str = "checkpoints"):
    # 1. Hardware detection (Crucial for Colab)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"🚀 Initializing Training on device: {device}")
    if device.type == 'cpu':
        logger.warning("WARNING: Training on CPU will be extremely slow. Please use a GPU.")

    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # 2. Load Data and Model
    logger.info("Loading Dataset...")
    dataset = VisioVoxDataset(metadata_path=metadata_path)
    
    # We use num_workers=2 in Colab to speed up data loading
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0) 

    logger.info("Loading VisioVox Model to GPU...")
    model = VisioVox().to(device)
    
    # 3. Optimizer and Loss Function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # L1 Loss is standard for spectrogram reconstruction (predicting the exact pixel/bin values)
    criterion = nn.L1Loss() 

    logger.info(f"Starting training loop for {epochs} epochs...")

    # 4. The Training Loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for batch_idx, (mixed_spec, target_spec, lips) in enumerate(dataloader):
            # Move tensors to GPU memory
            mixed_spec = mixed_spec.to(device)
            target_spec = target_spec.to(device)
            lips = lips.to(device)

            # Zero the gradients from the previous step
            optimizer.zero_grad()

            # Forward pass: The model predicts a mask (0 to 1)
            predicted_mask = model(mixed_spec, lips)

            # Apply the mask to the mixed audio to "filter" out the target speaker
            separated_spec = predicted_mask * mixed_spec

            # Calculate how far the separated audio is from the actual clean audio
            loss = criterion(separated_spec, target_spec)

            # Backward pass: Compute gradients and update weights
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Log progress every 10 batches
            if (batch_idx + 1) % 10 == 0:
                logger.info(f"Epoch [{epoch+1}/{epochs}] | Batch [{batch_idx+1}/{len(dataloader)}] | Loss: {loss.item():.4f}")

        # Epoch Summary
        avg_loss = running_loss / len(dataloader)
        logger.info(f"=== Epoch {epoch+1} Completed | Average Loss: {avg_loss:.4f} ===")

        # Save model weights after every epoch
        checkpoint_path = os.path.join(save_dir, f"visiovox_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        logger.info(f"💾 Saved checkpoint to: {checkpoint_path}")

if __name__ == "__main__":
    METADATA = "data/processed/dataset_metadata.csv"
    # Note: Batch size of 4 is safe for a 16GB T4 GPU. 
    train_model(METADATA, epochs=20, batch_size=4)