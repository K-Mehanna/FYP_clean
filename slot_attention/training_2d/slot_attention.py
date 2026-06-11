# Very similar to SlotAttention class from: https://github.com/UnheardChunk/SAA-CBR

import torch
import torch.nn as nn
from torch import Tensor

class SlotAttention(nn.Module):
    def __init__(
        self,
        input_dim: int = 64,
        num_slots: int = 7,
        slot_dim: int = 64,
        routing_iters: int = 3,
        hidden_dim: int = 128,
        temperature: float = 0.5,
    ):
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.routing_iters = routing_iters
        self.temperature = temperature

        self.ln_inputs = nn.LayerNorm(input_dim)
        self.ln_slots = nn.LayerNorm(self.slot_dim)

        self.W_q = nn.Parameter(torch.rand(self.slot_dim, self.slot_dim))
        self.W_k = nn.Parameter(torch.rand(input_dim, self.slot_dim))
        self.W_v = nn.Parameter(torch.rand(input_dim, self.slot_dim))
        self.loc = nn.Parameter(torch.zeros(1, self.slot_dim))
        self.logscale = nn.Parameter(torch.zeros(1, self.slot_dim))

        nn.init.xavier_uniform_(self.loc)
        nn.init.normal_(self.logscale, mean=0.0, std=0.5)

        self.gru = nn.GRUCell(self.slot_dim, self.slot_dim)
        self.mlp = nn.Sequential(
            nn.LayerNorm(self.slot_dim),
            nn.Linear(self.slot_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.slot_dim),
        )

        # Optional fixed seed for deterministic slot initialization at inference time.
        self._slot_init_seed: int | None = None
        self._train_slot_init_generator: torch.Generator | None = None

    def set_deterministic_slot_init_seed(self, seed: int) -> None:
        """Set a deterministic seed for slot initialization during inference.
        
        Args:
            seed: Random seed for the generator. When set, slot initialization will be deterministic.
        """
        self._slot_init_seed = seed
        self._train_slot_init_generator = None
    
    def forward(self, x: Tensor, num_slots: int | None = None) -> Tensor:
        # b: batch_size, n: num_inputs, c: input_dim, K: num_slots, d: slot_dim
        b = x.shape[0]
        # (b, n, c)
        x = self.ln_inputs(x)
        # (b, n, d)
        k = torch.einsum("bnc,cd->bnd", x, self.W_k)
        v = torch.einsum("bnc,cd->bnd", x, self.W_v)
        # (b, k, d)
        K = num_slots if num_slots is not None else self.num_slots
        
        # Use a seeded generator when deterministic slot init is enabled.
        # Training reuses a persistent generator so successive batches remain distinct
        # while still being reproducible across runs.
        if self._slot_init_seed is not None:
            if self.training:
                if self._train_slot_init_generator is None:
                    self._train_slot_init_generator = torch.Generator(device=x.device)
                    self._train_slot_init_generator.manual_seed(self._slot_init_seed)
                slot_gen = self._train_slot_init_generator
            else:
                slot_gen = torch.Generator(device=x.device)
                slot_gen.manual_seed(self._slot_init_seed)

            slots = self.loc + self.logscale.exp() * torch.randn(
                b, K, self.slot_dim, device=x.device, generator=slot_gen
            )
        else:
            slots = self.loc + self.logscale.exp() * torch.randn(
                b, K, self.slot_dim, device=x.device
            )

        for _ in range(self.routing_iters):
            slots_prev = slots
            slots = self.ln_slots(slots)
            # (b, k, d)
            q = torch.einsum("bkd,dd->bkd", slots, self.W_q)
            q = q * self.slot_dim**-0.5
            # (b, k, n)
            agreement = torch.einsum("bkd,bdn->bkn", q, k.transpose(-2, -1))
            # attn = agreement.softmax(dim=1) + 1e-8
            attn = (agreement / self.temperature).softmax(dim=1) + 1e-8
            attn = attn / attn.sum(dim=-1, keepdim=True)  # weighted mean
            # (b, k, d)
            updates = torch.einsum("bkn,bnd->bkd", attn, v)
            # (b*k, d)
            slots = self.gru(
                updates.reshape(-1, self.slot_dim),
                slots_prev.reshape(-1, self.slot_dim),
            )
            # (b, k, d)
            slots = slots.reshape(b, -1, self.slot_dim)
            slots = slots + self.mlp(slots)
        return slots


class PositionEmbed3D(nn.Module):
    def __init__(self, out_channels: int, resolution: tuple[int, int, int]):
        super().__init__()
        # (1, Depth, Height, Width, 6)
        self.register_buffer("grid", self.build_grid(resolution))
        self.mlp = nn.Linear(6, out_channels)  # 6 for (x, y, z, 1−x, 1−y, 1−z)

    def forward(self, x: Tensor) -> Tensor:
        # (1, D, H, W, out_channels)
        grid = self.mlp(self.grid)
        # Permute to (batch_size, out_channels, depth, height, width) to match x shape
        return x + grid.permute(0, 4, 1, 2, 3)

    def build_grid(self, resolution: tuple[int, int, int]) -> Tensor:
        ranges = [torch.linspace(0.0, 1.0, steps=r) for r in resolution]
        grid = torch.meshgrid(*ranges, indexing="ij")
        grid = torch.stack(grid, dim=-1)
        # Might need to replace two lines above with following
        # xx, yy, zz = torch.meshgrid(ranges, indexing="ij")
        # grid = torch.stack([xx, yy, zz], dim=-1)
        grid = grid.unsqueeze(0)
        return torch.cat([grid, 1.0 - grid], dim=-1)


class MLPHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.mlp_head = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, out_dim),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp_head(x)


class SlotClassifier3D(nn.Module):
    def __init__(
        self,
        in_shape: tuple[int, int, int, int] = (1, 64, 96, 96),     # (Channels, Depth, Height, Width)
        width: int = 128,
        num_slots: int = 5,         # Background, NCR/NET, ED, ET, and one extra for potential unknown objects
        slot_dim: int = 128,
        routing_iters: int = 10,
        temperature: float = 2.0,
        obj_info: dict = {
            "region_class": 4,  # Probability distribution across: Background, NCR/NET, ED, ET
            "coords": 3,        # 3D spatial coordinates of the region's centroid (x, y, z)
            "volume": 1,        # The normalized size/volume of the tumor region
        },
    ):
        super().__init__()
        # Swapped to Conv3D
        self.encoder = nn.Sequential(
            nn.Conv3d(in_shape[0], width, 5, padding=2),
            nn.ReLU(),
            nn.Conv3d(width, width, 5, padding=2),
            nn.ReLU(),
            nn.Conv3d(width, width, 5, padding=2),
            nn.ReLU(),
            nn.Conv3d(width, width, 5, padding=2),
            nn.ReLU(),
            PositionEmbed3D(width, in_shape[1:]),
        )

        self.mlp = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, width),
        )

        self.slot_attention = SlotAttention(
            input_dim=width,
            num_slots=num_slots,
            slot_dim=slot_dim,
            routing_iters=routing_iters,
            hidden_dim=width,
            temperature=temperature,
        )

        # 3D Slot Grid Size (Requires depth, height, width to be perfectly divisible by 16)
        self.slot_grid = (in_shape[1] // 16, in_shape[2] // 16, in_shape[3] // 16)
        # Swapped to ConvTranspose3d
        self.decoder = nn.Sequential(
            PositionEmbed3D(slot_dim, self.slot_grid),

            # nn.ConvTranspose3d(width, width, 5, stride=2, padding=2, output_padding=1),
            nn.ConvTranspose3d(width, width, 4, stride=2, padding=1),
            # nn.ConvTranspose3d(width, width, kernel_size=2, stride=2),
            # nn.Upsample(scale_factor=2, mode='nearest'),
            # nn.Conv3d(width, width, kernel_size=3, padding=1, padding_mode='replicate'),
            nn.LeakyReLU(),

            # nn.ConvTranspose3d(width, width, 5, stride=2, padding=2, output_padding=1),
            nn.ConvTranspose3d(width, width, 4, stride=2, padding=1),
            # nn.ConvTranspose3d(width, width, kernel_size=2, stride=2),
            # nn.Upsample(scale_factor=2, mode='nearest'),
            # nn.Conv3d(width, width, kernel_size=3, padding=1, padding_mode='replicate'),
            nn.LeakyReLU(),

            # nn.ConvTranspose3d(width, width, 5, stride=2, padding=2, output_padding=1),
            nn.ConvTranspose3d(width, width, 4, stride=2, padding=1),
            # nn.ConvTranspose3d(width, width, kernel_size=2, stride=2),
            # nn.Upsample(scale_factor=2, mode='nearest'),
            # nn.Conv3d(width, width, kernel_size=3, padding=1, padding_mode='replicate'),
            nn.LeakyReLU(),

            # nn.ConvTranspose3d(width, width, 5, stride=2, padding=2, output_padding=1),
            nn.ConvTranspose3d(width, width, 4, stride=2, padding=1),
            # nn.ConvTranspose3d(width, width, kernel_size=2, stride=2),
            # nn.Upsample(scale_factor=2, mode='nearest'),
            # nn.Conv3d(width, width, kernel_size=3, padding=1, padding_mode='replicate'),
            nn.LeakyReLU(),

            nn.ConvTranspose3d(width, width, 5, stride=1, padding=2),
            nn.LeakyReLU(),

            # Outputs 'channels + 1' (extra 1 for 3D volumetric alpha mask)
            nn.ConvTranspose3d(width, in_shape[0] + 1, 3, stride=1, padding=1),
        )

        self.mlp_heads = nn.ModuleList(
            [MLPHead(width, config) for config in list(obj_info.values())]
        )

    def set_deterministic_slot_init(self, seed: int) -> None:
        """Enable deterministic slot initialization during inference.
        
        Call this method before inference to ensure reproducible results.
        
        Args:
            seed: Random seed for slot initialization.
        
        Example:
            model.eval()
            model.set_deterministic_slot_init(seed=42)
            with torch.no_grad():
                output = model(x)
        """
        self.slot_attention.set_deterministic_slot_init_seed(seed)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        # b: batch_size, c: channels, d: depth, h: height, w: width
        b, c, d, h, w = x.shape
        
        # ENCODER (Runs completely in memory-saving float16)
        x = self.encoder(x)
        # (b, d*h*w, width)
        x_flat = self.mlp(x.reshape(*x.shape[:2], -1).permute(0, 2, 1))

        # Explicitly disable autocast for the routing and MLP heads
        # to protect the GRU and Softmax from microscopic float16 rounding errors.
        with torch.amp.autocast('cuda', enabled=False):

            # Cast the input to float32 before it enters the routing loop.
            # nan_to_num guards against float16 overflow (inf) in the encoder
            # propagating as NaN through LayerNorm into the slot attention.
            slots_fp32 = self.slot_attention(
                torch.nan_to_num(x_flat.float(), nan=0.0, posinf=1e4, neginf=-1e4),
            )

            # Run the MLP heads safely in float32
            batch_size, num_elements, input_size = slots_fp32.size()
            z = slots_fp32.view(-1, input_size)
            outputs = [head(z).view(batch_size, num_elements, -1) for head in self.mlp_heads]
            output = torch.cat(outputs, dim=2)

        # PyTorch automatically resumes float16 precision the moment this data hits the ConvTranspose3d
        
        # (b*num_slots, slot_dim, init_d, init_h, init_w)
        x_decode = slots_fp32.view(-1, slots_fp32.shape[-1])[:, :, None, None, None]
        x_decode = x_decode.repeat(1, 1, *self.slot_grid)
        
        # (b*num_slots, c + 1, d, h, w)
        x_decode = self.decoder(x_decode)

        # (b, num_slots, c + 1, d, h, w)
        x_decode = x_decode.view(b, -1, c + 1, d, h, w)
        
        # (b, num_slots, c, d, h, w), (b, num_slots, 1, d, h, w)
        recons, masks = torch.split(x_decode, [c, 1], dim=2)
        masks = masks.softmax(dim=1)
        
        # (b, c, d, h, w)
        recon_combined = torch.sum(recons * masks, dim=1)

        return recon_combined, recons, masks, slots_fp32, output