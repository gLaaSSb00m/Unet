import logging
import numpy as np
import torch
from PIL import Image
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import gc

def load_image(filename):
    """Load an image or tensor file into a PIL Image."""
    ext = splitext(filename)[1].lower()
    if ext == '.npy':
        return Image.fromarray(np.load(filename))
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy())
    else:
        return Image.open(filename)

class BasicDataset(Dataset):
    def __init__(self, images_dir: str, mask_dir: str, scale: float = 1.0, mask_suffix: str = '', max_images: int = None):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix

        # Load image IDs
        self.ids = [splitext(file)[0] for file in listdir(images_dir) if isfile(join(images_dir, file)) and not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
        if max_images is not None:
            self.ids = self.ids[:max_images]

        logging.info(f'Creating dataset with {len(self.ids)} examples')

        # Scan mask files to determine unique values
        logging.info('Scanning mask files to determine unique values')
        unique = []
        for idx in tqdm(self.ids, total=len(self.ids), desc="Scanning masks"):
            mask_file = list(self.mask_dir.glob(idx + self.mask_suffix + '.*'))
            if not mask_file:
                raise FileNotFoundError(f'No mask found for ID {idx}')
            mask = np.asarray(load_image(mask_file[0]).convert('L'))  # Convert to grayscale
            unique.append(np.unique(mask))  # Collect unique values
            # Free memory
            del mask
            gc.collect()

        self.mask_values = list(sorted(np.unique(np.concatenate(unique)).tolist()))
        logging.info(f'Unique mask values: {self.mask_values}')

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(mask_values, pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img = np.asarray(pil_img)

        if is_mask:
            mask = np.zeros((newH, newW), dtype=np.int64)
            for i, v in enumerate(mask_values):
                mask[img == v] = i
            return mask
        else:
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            else:
                img = img.transpose((2, 0, 1))
            if (img > 1).any():
                img = img / 255.0
            return img

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.mask_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        mask = load_image(mask_file[0]).convert('L')  # Convert mask to grayscale
        img = load_image(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(self.mask_values, img, self.scale, is_mask=False)
        mask = self.preprocess(self.mask_values, mask, self.scale, is_mask=True)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }

class CarvanaDataset(BasicDataset):
    def __init__(self, images_dir, mask_dir, scale=1, max_images=None):
        """
        Dataset for the Carvana image segmentation task (subclass of BasicDataset).

        Args:
            images_dir (str): Directory containing input images.
            mask_dir (str): Directory containing mask images.
            scale (float): Scale factor for resizing images.
            max_images (int, optional): Maximum number of images to load.
        """
        super().__init__(images_dir, mask_dir, scale, mask_suffix='_mask', max_images=max_images)

def train_model(
    model,
    device,
    epochs=10,
    batch_size=10,
    learning_rate=1e-4,
    img_scale=0.125,
    val_percent=0.1,
):
    # 1. Load the dataset
    dataset = BasicDataset(dir_img, dir_mask, scale=img_scale)
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

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

        with tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}/{epochs}', unit='batch') as pbar:
            for batch in train_loader:
                images, true_masks = batch['image'], batch['mask']
                images = images.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.long)
                true_masks = true_masks.squeeze(1)

                # Forward pass
                masks_pred = model(images)
                loss = criterion(masks_pred, true_masks)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Update metrics
                epoch_loss += loss.item()
                preds = masks_pred.argmax(dim=1) if model.n_classes > 1 else (torch.sigmoid(masks_pred) > 0.5).float()
                correct_train += (preds == true_masks).sum().item()
                total_train += true_masks.numel()

                pbar.update(1)
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

    return train_losses, train_accuracies, val_losses, val_accuracies