<<<<<<< HEAD
import torch
from unet import UNet as _UNet

def unet_carvana(pretrained=False, scale=0.5):
    """
    UNet model trained on the Carvana dataset ( https://www.kaggle.com/c/carvana-image-masking-challenge/data ).
    Set the scale to 0.5 (50%) when predicting.
    """
    net = _UNet(n_channels=3, n_classes=2, bilinear=False)
    if pretrained:
        if scale == 0.5:
            checkpoint = 'https://github.com/milesial/Pytorch-UNet/releases/download/v3.0/unet_carvana_scale0.5_epoch2.pth'
        elif scale == 1.0:
            checkpoint = 'https://github.com/milesial/Pytorch-UNet/releases/download/v3.0/unet_carvana_scale1.0_epoch2.pth'
        else:
            raise RuntimeError('Only 0.5 and 1.0 scales are available')
        state_dict = torch.hub.load_state_dict_from_url(checkpoint, progress=True)
        if 'mask_values' in state_dict:
            state_dict.pop('mask_values')
        net.load_state_dict(state_dict)

    return net

=======
import torch
from unet import UNet as _UNet

def unet_carvana(pretrained=False, scale=0.5):
    """
    UNet model trained on the Carvana dataset ( https://www.kaggle.com/c/carvana-image-masking-challenge/data ).
    Set the scale to 0.5 (50%) when predicting.
    """
    net = _UNet(n_channels=3, n_classes=2, bilinear=False)
    if pretrained:
        if scale == 0.5:
            checkpoint = 'https://github.com/milesial/Pytorch-UNet/releases/download/v3.0/unet_carvana_scale0.5_epoch2.pth'
        elif scale == 1.0:
            checkpoint = 'https://github.com/milesial/Pytorch-UNet/releases/download/v3.0/unet_carvana_scale1.0_epoch2.pth'
        else:
            raise RuntimeError('Only 0.5 and 1.0 scales are available')
        state_dict = torch.hub.load_state_dict_from_url(checkpoint, progress=True)
        if 'mask_values' in state_dict:
            state_dict.pop('mask_values')
        net.load_state_dict(state_dict)

    return net

>>>>>>> cb2ce052e02f2192550c7ec6996f9db86bdc3145
