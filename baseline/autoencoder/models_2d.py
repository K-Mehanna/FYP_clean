"""
2D counterparts of the 3D autoencoder models in models.py.

Three classes mirror the 3D versions exactly:
  ResNetSpatialEncoder2D  — torchvision ResNet18/50, layer4 stride patched to 1
  ResNetAutoencoder2D     — Spatial encoder + 2D convolutional decoder
  ResNetClassifier2D      — Frozen encoder + adaptive pooling + trainable head

Architecture note:
  Standard torchvision ResNet applies stride=2 in layer4, giving 32× total spatial
  reduction. We patch the first block of layer4 back to stride=1 (matching the MONAI
  3D encoder used in models.py), so the total reduction is 16× — identical to the 3D
  counterpart. For 240×240 input this gives a 15×15 bottleneck; 4 ConvTranspose2d
  blocks with stride=2 recover 240×240 exactly.
"""

import torch
import torch.nn as nn
import torchvision.models as tvm
from torchvision.models.resnet import BasicBlock, Bottleneck


def _patch_layer4_stride(layer4: nn.Sequential) -> None:
    """Change the first block of layer4 from stride=2 to stride=1 in-place."""
    block = layer4[0]
    if isinstance(block, BasicBlock):
        block.conv1.stride = (1, 1)
    elif isinstance(block, Bottleneck):
        block.conv2.stride = (1, 1)
    else:
        raise TypeError(f"Unexpected block type in layer4: {type(block)}")
    if block.downsample is not None:
        block.downsample[0].stride = (1, 1)


class ResNetSpatialEncoder2D(nn.Module):
    """
    2D spatial feature extractor built on torchvision ResNet18/50.

    Layer4's stride is patched to 1 so the total spatial reduction matches the
    3D MONAI encoder (16× instead of 32×). For 240×240 input, the bottleneck is
    (B, out_channels, 15, 15).
    """

    _OUT_CHANNELS = {"resnet18": 512, "resnet50": 2048}

    def __init__(self, model_name: str = "resnet18", in_channels: int = 5):
        super().__init__()

        if model_name not in self._OUT_CHANNELS:
            raise ValueError(f"model_name must be 'resnet18' or 'resnet50', got '{model_name}'")

        base = tvm.resnet18(weights=None) if model_name == "resnet18" else tvm.resnet50(weights=None)

        # Replace conv1 to accept in_channels inputs (default 5 quantile slices)
        base.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Patch layer4 stride from 2 → 1 (mirrors MONAI 3D encoder's 16× reduction)
        _patch_layer4_stride(base.layer4)

        # Keep only spatial feature layers; discard avgpool and fc
        self.conv1   = base.conv1
        self.bn1     = base.bn1
        self.relu    = base.relu
        self.maxpool = base.maxpool
        self.layer1  = base.layer1
        self.layer2  = base.layer2
        self.layer3  = base.layer3
        self.layer4  = base.layer4

        self.out_channels = self._OUT_CHANNELS[model_name]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input:  (B, in_channels, H, W)
        Output: (B, out_channels, H/16, W/16)
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class ResNetAutoencoder2D(nn.Module):
    """
    Self-supervised autoencoder for pre-training on BraTS PNG data.

    The decoder mirrors the 3D version: 4 ConvTranspose2d blocks each doubling
    spatial dimensions, followed by a final Conv2d that maps back to in_channels.
    """

    def __init__(self, model_name: str = "resnet18", in_channels: int = 5):
        super().__init__()
        self.in_channels = in_channels
        self.encoder = ResNetSpatialEncoder2D(model_name=model_name, in_channels=in_channels)

        in_c = self.encoder.out_channels
        self.decoder_conv = nn.Sequential(
            # (15,15) → (30,30)
            nn.ConvTranspose2d(in_c, 256, kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # (30,30) → (60,60)
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # (60,60) → (120,120)
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # (120,120) → (240,240)
            nn.ConvTranspose2d(64, 16, kernel_size=2, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            # Map back to in_channels
            nn.Conv2d(16, in_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)           # (B, out_channels, H/16, W/16)
        x_hat = self.decoder_conv(z)  # (B, in_channels, H, W)
        return x_hat, z


class ResNetClassifier2D(nn.Module):
    """
    2D classifier: frozen pre-trained encoder + adaptive pooling + MLP head.

    Mirrors ResNetClassifier exactly (same frozen/unfreeze logic, same head
    architecture) but operates on 2D feature maps from ResNetSpatialEncoder2D.
    """

    def __init__(
        self,
        autoencoder: ResNetAutoencoder2D,
        num_classes: int,
        dropout: float = 0.3,
        unfreeze_encoder: bool = False,
    ):
        super().__init__()

        self.encoder = autoencoder.encoder

        # Pool (H/16, W/16) spatial tensor to a 1D vector
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

        self.classifier = nn.Sequential(
            nn.Linear(self.encoder.out_channels, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(64, num_classes),
        )

        self._encoder_frozen = False
        self.set_encoder_frozen(not unfreeze_encoder)

    def set_encoder_frozen(self, frozen: bool) -> None:
        for param in self.encoder.parameters():
            param.requires_grad = not frozen
        self._encoder_frozen = frozen
        if frozen:
            self.encoder.eval()
        else:
            self.encoder.train()
        state = "frozen" if frozen else "unfrozen"
        print(f"[ResNetClassifier2D] Encoder is {state}.")

    def train(self, mode: bool = True):
        super().train(mode)
        if self._encoder_frozen:
            self.encoder.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._encoder_frozen:
            with torch.no_grad():
                spatial_z = self.encoder(x)
        else:
            spatial_z = self.encoder(x)

        z = self.pool(spatial_z)   # (B, out_channels, 1, 1)
        z = self.flatten(z)        # (B, out_channels)
        return self.classifier(z)
