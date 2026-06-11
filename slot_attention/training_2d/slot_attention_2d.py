import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from slot_attention.training_2d.slot_attention import SlotAttention, MLPHead


class SlotClassifier2DSkip(nn.Module):
    """U-Net variant with encoder skip connections used in checkpoints after the
    sequential encoder/decoder refactor (e.g. v10_ri10).  Load these checkpoints
    with this class instead of SlotClassifier2D."""

    def __init__(
        self,
        in_shape: tuple[int, int, int] = (1, 240, 240),
        width: int = 64,
        num_slots: int = 5,
        slot_dim: int = 64,
        routing_iters: int = 7,
        temperature: float = 0.5,
        obj_info: dict = {
            "region_class": 4,
            "coords": 2,
            "volume": 1,
        },
    ):
        super().__init__()
        c, h, w = in_shape
        assert h % 16 == 0 and w % 16 == 0

        self.enc1 = nn.Sequential(nn.Conv2d(c, width, 5, stride=2, padding=2), nn.ReLU())
        self.enc2 = nn.Sequential(nn.Conv2d(width, width, 5, stride=2, padding=2), nn.ReLU())
        self.enc3 = nn.Sequential(nn.Conv2d(width, width, 5, stride=2, padding=2), nn.ReLU())
        self.enc4 = nn.Sequential(nn.Conv2d(width, width, 5, stride=2, padding=2), nn.ReLU())
        self.pos_embed_enc = PositionEmbed2D(width, (h // 16, w // 16))

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

        self.slot_grid = (h // 16, w // 16)
        self.dec_pos_embed = PositionEmbed2D(slot_dim, self.slot_grid)

        self.up1 = nn.Sequential(nn.ConvTranspose2d(width, width, 4, stride=2, padding=1), nn.LeakyReLU())
        self.up2 = nn.Sequential(nn.ConvTranspose2d(width, width, 4, stride=2, padding=1), nn.LeakyReLU())
        self.up3 = nn.Sequential(nn.ConvTranspose2d(width, width, 4, stride=2, padding=1), nn.LeakyReLU())
        self.up4 = nn.Sequential(nn.ConvTranspose2d(width, width, 4, stride=2, padding=1), nn.LeakyReLU())
        self.refine1 = nn.Sequential(nn.Conv2d(width, width, 5, padding=2), nn.LeakyReLU())
        self.out_conv = nn.ConvTranspose2d(width, c + 1, 3, padding=1)

        self.skip_conv3 = nn.Conv2d(width * 2, width, 1)
        self.skip_conv2 = nn.Conv2d(width * 2, width, 1)
        self.skip_conv1 = nn.Conv2d(width * 2, width, 1)

        self.mlp_heads = nn.ModuleList(
            [MLPHead(slot_dim, config) for config in list(obj_info.values())]
        )

        self._c = c
        self._h = h
        self._w = w

    def set_deterministic_slot_init(self, seed: int) -> None:
        self.slot_attention.set_deterministic_slot_init_seed(seed)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        b, c, h, w = x.shape

        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        x4 = self.pos_embed_enc(x4)

        x_flat = self.mlp(x4.reshape(b, x4.shape[1], -1).permute(0, 2, 1))

        with torch.amp.autocast('cuda', enabled=False):
            slots = self.slot_attention(
                torch.nan_to_num(x_flat.float(), nan=0.0, posinf=1e4, neginf=-1e4)
            )
            batch_size, num_elements, input_size = slots.size()
            z = slots.view(-1, input_size)
            outputs = [head(z).view(batch_size, num_elements, -1) for head in self.mlp_heads]
            output = torch.cat(outputs, dim=2)

        with torch.amp.autocast('cuda', enabled=False):
            num_slots = slots.shape[1]
            x_decode = slots.view(-1, slots.shape[-1])[:, :, None, None]
            x_decode = x_decode.repeat(1, 1, *self.slot_grid)
            x_decode = self.dec_pos_embed(x_decode)

            d = self.up1(x_decode)
            x3_exp = x3.unsqueeze(1).expand(-1, num_slots, -1, -1, -1).reshape(b * num_slots, -1, *x3.shape[2:])
            d = self.skip_conv3(torch.cat([d, x3_exp], dim=1))

            d = self.up2(d)
            x2_exp = x2.unsqueeze(1).expand(-1, num_slots, -1, -1, -1).reshape(b * num_slots, -1, *x2.shape[2:])
            d = self.skip_conv2(torch.cat([d, x2_exp], dim=1))

            d = self.up3(d)
            x1_exp = x1.unsqueeze(1).expand(-1, num_slots, -1, -1, -1).reshape(b * num_slots, -1, *x1.shape[2:])
            d = self.skip_conv1(torch.cat([d, x1_exp], dim=1))

            d = self.up4(d)
            d = self.refine1(d)
            x_decode = self.out_conv(d)

            x_decode = x_decode.view(b, -1, c + 1, h, w)
            recons, masks = torch.split(x_decode, [c, 1], dim=2)
            masks = masks.softmax(dim=1)
            recon_combined = torch.sum(recons * masks, dim=1)

        return recon_combined, recons, masks, slots, output

class PositionEmbed2D(nn.Module):
    def __init__(self, out_channels: int, resolution: tuple[int, int]):
        super().__init__()
        # (1, H, W, 4)
        self.register_buffer("grid", self.build_grid(resolution))
        self.mlp = nn.Linear(4, out_channels)  # 4 for (x, y, 1-x, 1-y)

    def forward(self, x: Tensor) -> Tensor:
        # (1, H, W, out_channels) -> (1, out_channels, H, W)
        grid = self.mlp(self.grid).permute(0, 3, 1, 2)
        return x + grid

    def build_grid(self, resolution: tuple[int, int]) -> Tensor:
        ranges = [torch.linspace(0.0, 1.0, steps=r) for r in resolution]
        grid = torch.meshgrid(*ranges, indexing="ij")
        grid = torch.stack(grid, dim=-1).unsqueeze(0)  # (1, H, W, 2)
        return torch.cat([grid, 1.0 - grid], dim=-1)   # (1, H, W, 4)


class SlotClassifier2D(nn.Module):
    def __init__(
        self,
        in_shape: tuple[int, int, int] = (1, 240, 240),   # (Channels, Height, Width)
        width: int = 64,
        num_slots: int = 5,
        slot_dim: int = 64,
        routing_iters: int = 7,
        temperature: float = 0.5,
        encoder_depth: int = 4,
        enc3_init_skip: bool = False,
        use_mask_pool_classifier: bool = False,
        obj_info: dict = {
            "region_class": 4,  # Background, NCR/NET, ED, ET
            "coords": 2,        # 2D centroid (y, x) normalised to [0, 1]
            "volume": 1,        # Normalised pixel area
        },
    ):
        super().__init__()
        c, h, w = in_shape
        stride = 2 ** encoder_depth
        assert h % stride == 0 and w % stride == 0, \
            f"H and W must be divisible by {stride} for encoder_depth={encoder_depth}."
        assert encoder_depth in (3, 4), "encoder_depth must be 3 or 4."
        self._encoder_depth = encoder_depth
        self._enc3_init_skip = enc3_init_skip and (encoder_depth == 3)

        # Encoder: named blocks to expose intermediate feature maps for skip connections.
        # Each block halves spatial dims
        self.enc1 = nn.Sequential(nn.Conv2d(c, width, 5, stride=2, padding=2), nn.ReLU())      # → 120×120
        self.enc2 = nn.Sequential(nn.Conv2d(width, width, 5, stride=2, padding=2), nn.ReLU())  # →  60×60
        self.enc3 = nn.Sequential(nn.Conv2d(width, width, 5, stride=2, padding=2), nn.ReLU())  # →  30×30
        if encoder_depth == 4:
            self.enc4 = nn.Sequential(nn.Conv2d(width, width, 5, stride=2, padding=2), nn.ReLU())  # → 15×15

        # Slot grid resolution depends on encoder depth.
        self.slot_grid = (h // stride, w // stride)
        self.pos_embed_enc = PositionEmbed2D(width, self.slot_grid)

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

        self._num_slots = num_slots

        # Decoder: individual named layers so skip features can be injected between steps.
        self.dec_pos_embed = PositionEmbed2D(slot_dim, self.slot_grid)
        self.up1     = nn.Sequential(nn.ConvTranspose2d(width, width, 4, stride=2, padding=1), nn.LeakyReLU())
        self.up2     = nn.Sequential(nn.ConvTranspose2d(width, width, 4, stride=2, padding=1), nn.LeakyReLU())
        self.up3     = nn.Sequential(nn.ConvTranspose2d(width, width, 4, stride=2, padding=1), nn.LeakyReLU())
        if encoder_depth == 4:
            self.up4 = nn.Sequential(nn.ConvTranspose2d(width, width, 4, stride=2, padding=1), nn.LeakyReLU())
        self.refine1 = nn.Sequential(nn.ConvTranspose2d(width, width, 5, stride=1, padding=2), nn.LeakyReLU())
        self.out_conv = nn.ConvTranspose2d(width, c + 1, 3, stride=1, padding=1)

        # U-Net skip connections: 1×1 convs reduce channels after concatenation
        if encoder_depth == 4:
            self.skip_conv3 = nn.Conv2d(2 * width, width, 1)
        if self._enc3_init_skip:
            self.skip_conv_init = nn.Conv2d(2 * width, width, 1)
        self.skip_conv2 = nn.Conv2d(2 * width, width, 1)
        self.skip_conv1 = nn.Conv2d(2 * width, width, 1)

        self.mlp_heads = nn.ModuleList(
            [MLPHead(slot_dim, config) for config in list(obj_info.values())]
        )

        # Optional mask-pooled classifier
        self._use_mask_pool_classifier = use_mask_pool_classifier
        if use_mask_pool_classifier:
            pool_in = slot_dim + width
            self.pool_class_head = nn.Sequential(
                nn.LayerNorm(pool_in),
                nn.Linear(pool_in, width),
                nn.ReLU(),
                nn.Linear(width, 4),
            )

        self._c = c
        self._h = h
        self._w = w

    def set_deterministic_slot_init(self, seed: int) -> None:
        self.slot_attention.set_deterministic_slot_init_seed(seed)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        b, c, h, w = x.shape

        # Encoder runs in fp16 under AMP; save intermediate maps for skip connections.
        s1 = self.enc1(x)    # (b, width, 120, 120)
        s2 = self.enc2(s1)   # (b, width,  60,  60)
        s3 = self.enc3(s2)   # (b, width,  30,  30)

        if self._encoder_depth == 4:
            s4  = self.enc4(s3)          # (b, width, 15, 15)
            enc = self.pos_embed_enc(s4)
        else:
            enc = self.pos_embed_enc(s3) # (b, width, 30, 30)

        # Flatten spatial dims: (b, h*w, width)
        x_flat = self.mlp(enc.reshape(b, enc.shape[1], -1).permute(0, 2, 1))

        # Slot attention and MLP heads run in fp32: softmax and GRU are NaN-prone in fp16.
        # nan_to_num guards against fp16 overflow (inf) from the encoder propagating as NaN.
        with torch.amp.autocast('cuda', enabled=False):
            slots = self.slot_attention(
                torch.nan_to_num(x_flat.float(), nan=0.0, posinf=1e4, neginf=-1e4)
            )

            batch_size, num_elements, input_size = slots.size()
            z = slots.view(-1, input_size)
            outputs = [head(z).view(batch_size, num_elements, -1) for head in self.mlp_heads]
            output = torch.cat(outputs, dim=2)

        # Decoder must run in fp32
        with torch.amp.autocast('cuda', enabled=False):
            K = slots.shape[1]
            x_decode = slots.view(-1, slots.shape[-1])[:, :, None, None]
            x_decode = x_decode.repeat(1, 1, *self.slot_grid)

            # Broadcast a batch encoder feature map to (b*K, C, H', W') for concatenation.
            def _expand(feat: Tensor) -> Tensor:
                return feat.float().unsqueeze(1).expand(-1, K, -1, -1, -1).reshape(b * K, *feat.shape[1:])

            d = self.dec_pos_embed(x_decode)

            if self._encoder_depth == 4:
                # 15→30, skip enc3; 30→60, skip enc2; 60→120, skip enc1; 120→240
                d = self.up1(d)
                d = self.skip_conv3(torch.cat([d, _expand(s3)], dim=1))
                d = self.up2(d)
                d = self.skip_conv2(torch.cat([d, _expand(s2)], dim=1))
                d = self.up3(d)
                d = self.skip_conv1(torch.cat([d, _expand(s1)], dim=1))
                d = self.up4(d)
            else:
                # 30→60, skip enc2; 60→120, skip enc1; 120→240
                # Optional enc3 initial skip at 30×30 before first upsampling.
                if self._enc3_init_skip:
                    d = self.skip_conv_init(torch.cat([d, _expand(s3)], dim=1))
                d = self.up1(d)
                d = self.skip_conv2(torch.cat([d, _expand(s2)], dim=1))
                d = self.up2(d)
                d = self.skip_conv1(torch.cat([d, _expand(s1)], dim=1))
                d = self.up3(d)

            d = self.refine1(d)
            x_decode = self.out_conv(d)                    # (b*K, c+1, 240, 240)

            # (b, num_slots, c+1, h, w)
            x_decode = x_decode.view(b, -1, c + 1, h, w)

            recons, masks = torch.split(x_decode, [c, 1], dim=2)
            masks = masks.softmax(dim=1)

            # (b, c, h, w)
            recon_combined = torch.sum(recons * masks, dim=1)

            if self._use_mask_pool_classifier:
                # Pool enc3 (always 30×30) per slot using decoder soft masks, then
                # classify from [slot_vec ‖ pooled_enc3]. Replaces slot-vector-only
                # class logits (output dims 0–3) while leaving dims 4–6 unchanged.
                pool_h = s3.shape[2]
                masks_small = F.interpolate(
                    masks.squeeze(2).reshape(b * K, 1, h, w),
                    size=(pool_h, pool_h), mode='bilinear', align_corners=False,
                ).reshape(b, K, 1, pool_h, pool_h)
                enc3_exp = s3.float().unsqueeze(1).expand(b, K, -1, pool_h, pool_h)
                mask_w   = masks_small.sum(dim=(3, 4)).clamp(min=1e-6)
                pooled   = (enc3_exp * masks_small).sum(dim=(3, 4)) / mask_w
                pool_cls = self.pool_class_head(
                    torch.cat([slots.float(), pooled], dim=-1)
                )
                output = torch.cat([pool_cls, output[..., 4:]], dim=-1)

        return recon_combined, recons, masks, slots, output


class SlotClassifier2DShallow(SlotClassifier2D):
    """SlotClassifier2D with encoder_depth=3: 3-level encoder/decoder so the slot
    grid is 30×30 (vs 15×15) for 240×240 inputs. Use this class for checkpoints
    trained with the 3-level architecture."""

    def __init__(
        self,
        in_shape: tuple[int, int, int] = (1, 240, 240),
        width: int = 64,
        num_slots: int = 5,
        slot_dim: int = 64,
        routing_iters: int = 7,
        temperature: float = 0.5,
        obj_info: dict = {
            "region_class": 4,
            "coords": 2,
            "volume": 1,
        },
    ):
        super().__init__(
            in_shape=in_shape,
            width=width,
            num_slots=num_slots,
            slot_dim=slot_dim,
            routing_iters=routing_iters,
            temperature=temperature,
            encoder_depth=3,
            obj_info=obj_info,
        )
