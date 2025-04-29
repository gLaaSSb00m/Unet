import torch
import logging
from pathlib import Path
from torch.utils.data import DataLoader, random_split
from torch import optim
from tqdm import tqdm
from unet.unet_model import UNet
from utils.data_loading import BasicDataset
import matplotlib.pyplot as plt

# Paths to the dataset
dir_img = Path('segmentation_full_body_mads_dataset_1192_img/images')
dir_mask = Path('segmentation_full_body_mads_dataset_1192_img/masks')
dir_checkpoint = Path('./checkpoints/')

# Training function
def train_model(
    model,
    device,
    epochs=10,
    batch_size=1,  # Reduced to 1 to prevent memory issues
    learning_rate=1e-4,
    img_scale=0.125,
    val_percent=0.1,
):
    # 1. Load the dataset
    dataset = BasicDataset(dir_img, dir_mask, scale=img_scale)
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)

    # 2. Set up optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss() if model.n_classes > 1 else torch.nn.BCEWithLogitsLoss()

    # Initialize lists to store metrics
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    # 3. Training loop
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        correct_train = 0
        total_train = 0

        with tqdm(total=len(train_set), desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks = batch['image'], batch['mask']
                images = images.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.long)
                true_masks = true_masks.squeeze(1)

                # Normalize mask values to valid range
                true_masks = true_masks % model.n_classes

                # Forward pass
                masks_pred = model(images)
                loss = criterion(masks_pred, true_masks)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Update training metrics
                epoch_loss += loss.item()
                preds = masks_pred.argmax(dim=1) if model.n_classes > 1 else (torch.sigmoid(masks_pred) > 0.5).float()
                correct_train += (preds == true_masks).sum().item()
                total_train += true_masks.numel()

                pbar.update(images.shape[0])
                pbar.set_postfix(**{'loss (batch)': loss.item()})

        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        train_accuracy = correct_train / total_train
        train_accuracies.append(train_accuracy)

        # Validation loop
        model.eval()
        val_loss = 0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for batch in val_loader:
                images, true_masks = batch['image'], batch['mask']
                images = images.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.long)
                true_masks = true_masks.squeeze(1)

                true_masks = true_masks % model.n_classes

                masks_pred = model(images)
                loss = criterion(masks_pred, true_masks)
                val_loss += loss.item()

                preds = masks_pred.argmax(dim=1) if model.n_classes > 1 else (torch.sigmoid(masks_pred) > 0.5).float()
                correct_val += (preds == true_masks).sum().item()
                total_val += true_masks.numel()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        val_accuracy = correct_val / total_val
        val_accuracies.append(val_accuracy)

        # Log metrics
        logging.info(
            f'Epoch {epoch + 1}: '
            f'Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, '
            f'Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}'
        )

        # Save checkpoint
        dir_checkpoint.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), str(dir_checkpoint / f'checkpoint_epoch{epoch + 1}.pth'))
        logging.info(f'Checkpoint {epoch + 1} saved!')

    return train_losses, train_accuracies, val_losses, val_accuracies

# Main function
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # Check if GPU is available
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logging.info(f"GPU is available: {torch.cuda.get_device_name(0)}")
        torch.cuda.empty_cache()  # Clear GPU memory
    else:
        device = torch.device('cpu')
        logging.info("GPU is not available. Using CPU.")

    # Define the model
    n_channels = 3
    n_classes = 2
    model = UNet(n_channels=n_channels, n_classes=n_classes, bilinear=True)
    model.to(device=device)

    # Train the model
    train_losses, train_accuracies, val_losses, val_accuracies = train_model(
        model=model,
        device=device,
        epochs=10,
        batch_size=1,  # Reduced to 1 to prevent memory issues
        learning_rate=1e-4,
        img_scale=0.125,
        val_percent=0.1,
    )

    # Plot the loss and accuracy curves
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curve')
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label='Training Accuracy')
    plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy Curve')
    plt.legend()
    plt.grid()
    plt.show()