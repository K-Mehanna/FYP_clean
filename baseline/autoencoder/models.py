"""
Three classes:
  ResNetSpatialEncoder: MONAI ResNet18/50 modified to return the raw 4D spatial feature map
  ResNetAutoencoder: Spatial Encoder + 3D convolutional decoder
  ResNetClassifier: Frozen encoder + Adaptive Pooling + trainable classification head
"""

import torch
import torch.nn as nn
from monai.networks.nets import resnet18, resnet50

class ResNetSpatialEncoder(nn.Module):
    """
    Wraps MONAI 3D ResNet 18 or 50 to act as a purely spatial feature extractor.
    Mmanually strip out the final AdaptiveAvgPool3d and Linear layers so 
    the network retains its 3D spatial priors for reconstruction
    """
    _RAW_DIM = {"resnet18": 512, "resnet50": 2048}

    def __init__(self, model_name: str = "resnet18"):
        super().__init__()

        if model_name not in self._RAW_DIM:
            raise ValueError(f"model_name must be 'resnet18' or 'resnet50', got '{model_name}'")

        builder = resnet18 if model_name == "resnet18" else resnet50
        
        # Instantiate full MONAI model without FC layer
        full_model = builder(
            spatial_dims=3,
            n_input_channels=1,
            feed_forward=False
        )
        
        # Extract only spatial feature layers, discarding the final pooling layer
        self.conv1   = full_model.conv1
        self.bn1     = full_model.bn1
        self.act     = full_model.act
        self.maxpool = full_model.maxpool
        self.layer1  = full_model.layer1
        self.layer2  = full_model.layer2
        self.layer3  = full_model.layer3
        self.layer4  = full_model.layer4
        
        self.out_channels = self._RAW_DIM[model_name]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input:  (B, 1, 64, 96, 96)
        Output: (B, 1, 2, 3, 3)
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

class ResNetAutoencoder(nn.Module):
    """
    Autoencoder for self-supervised pre-training on BraTS
    """
    def __init__(self, model_name: str = "resnet18"):
        super().__init__()

        # Encoder
        self.encoder = ResNetSpatialEncoder(
            model_name=model_name,
        )

        # Decoder
        # kernel_size=2, stride=2 doubles the spatial dimensions at each step.
        in_c = self.encoder.out_channels
        
        self.decoder_conv = nn.Sequential(
            # (4,6,6) -> (8,12,12)
            nn.ConvTranspose3d(in_c, 256, kernel_size=2, stride=2),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),

            # (8,12,12) -> (16,24,24)
            nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),

            # (16,24,24) -> (32,48,48)
            nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),

            # (32,48,48) -> (64,96,96)
            nn.ConvTranspose3d(64, 16, kernel_size=2, stride=2),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),

            # Final reconstruction layer mapping to 1 channel
            nn.Conv3d(16, 1, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)          # (B, Channels, 2, 3, 3)
        x_hat = self.decoder_conv(z) # (B, 1, 64, 96, 96)
        return x_hat, z


class ResNetClassifier(nn.Module):
    """
    Applies pooling step that was stripped from the Autoencoder phase, 
    and securely freezes the batch statistics
    """
    def __init__(
        self,
        autoencoder: ResNetAutoencoder,
        num_classes: int,
        dropout: float = 0.3,
        unfreeze_encoder: bool = False,
    ):
        super().__init__()

        self.encoder = autoencoder.encoder
        
        # Pool (2,3,3) spatial tensor back into 1D vector for classification
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.flatten = nn.Flatten()

        self.classifier = nn.Sequential(
            nn.Linear(self.encoder.out_channels, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(64, num_classes),
        )

        # Freeze state initialisation
        self._encoder_frozen = False
        self.set_encoder_frozen(not unfreeze_encoder)

    def set_encoder_frozen(self, frozen: bool):
        """Freeze or unfreeze the encoder, explicitly managing eval() mode for BatchNorm."""
        for param in self.encoder.parameters():
            param.requires_grad = not frozen
            
        self._encoder_frozen = frozen
        
        if frozen:
            self.encoder.eval()  # Lock BatchNorm statistics
        else:
            self.encoder.train()
            
        state = "frozen" if frozen else "unfrozen"
        print(f"[ResNetClassifier] Encoder is {state}.")

    def train(self, mode: bool = True):
        """
        Override PyTorch's default train() to prevent global training loop 
        from un-evaling the frozen encoder and ruining BatchNorm stats.
        """
        super().train(mode)
        if self._encoder_frozen:
            self.encoder.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Context manager ensures no gradients are tracked when frozen
        if self._encoder_frozen:
            with torch.no_grad():
                spatial_z = self.encoder(x) # (B, C, 2, 3, 3)
        else:
            spatial_z = self.encoder(x)

        z = self.pool(spatial_z)            # (B, C, 1, 1, 1)
        z = self.flatten(z)                 # (B, C)
        
        return self.classifier(z)