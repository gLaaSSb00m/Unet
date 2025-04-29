from unet import UNet
from torchsummary import summary
import torch

# Define the model
n_channels = 3  # Example: 3 for RGB images
n_classes = 2   # Example: 2 for binary segmentation
bilinear = False  # Use bilinear upsampling or not
model = UNet(n_channels, n_classes, bilinear)

# Move the model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Print the model summary
input_size = (n_channels, 256, 256)  # Example input size (C, H, W)
summary(model, input_size)