import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from typing import Optional
from einops import rearrange


def drop_path_f(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path_f(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


def window_partition(x, window_size: int):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size: int, H: int, W: int):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=(self.window_size, self.window_size), num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, attn_mask):
        H, W = self.H, self.W
        B, L, C = x.shape

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
            attn_mask = None

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        attn_windows = self.attn(x_windows, mask=attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class BasicLayer(nn.Module):
    def __init__(self, dim, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.window_size = window_size
        self.use_checkpoint = use_checkpoint
        self.shift_size = window_size // 2

        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else self.shift_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)])

        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def create_mask(self, x, H, W):
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x, H, W):
        attn_mask = self.create_mask(x, H, W)
        for blk in self.blocks:
            blk.H, blk.W = H, W
            if not torch.jit.is_scripting() and self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, attn_mask)
            else:
                x = blk(x, attn_mask)
        if self.downsample is not None:
            x = self.downsample(x, H, W)
            H, W = (H + 1) // 2, (W + 1) // 2
        return x, H, W


class PatchMerging(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W):
        B, L, C = x.shape
        x = x.view(B, H, W, C)

        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        H = H // 2
        W = W // 2
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, -1, 4 * C)

        x = self.norm(x)
        x = self.reduction(x)
        return x, H, W


class PatchEmbed(nn.Module):

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=nn.LayerNorm):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        _, _, H, W = x.shape
        x = self.proj(x)
        Ho, Wo = H // self.patch_size, W // self.patch_size
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, Ho, Wo


class StandardPatchExpanding(nn.Module):

    def __init__(self, input_dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_dim = input_dim
        # 将维度放大2倍，然后在空间上展开为 2x2
        self.expand = nn.Linear(input_dim, 2 * input_dim, bias=False)
        self.norm = norm_layer(input_dim // 2)

    def forward(self, x, H, W):
        B, L, C = x.shape
        x = self.expand(x)  # [B, H*W, 2*C]
        x = rearrange(x, 'b (h w) (c p1 p2) -> b (h p1) (w p2) c', h=H, w=W, p1=2, p2=2)
        x = rearrange(x, 'b h w c -> b (h w) c')
        x = self.norm(x)
        return x, H * 2, W * 2


class FinalPatchExpanding(nn.Module):
    """最后一层还原到原始分辨率 (4倍放大)"""

    def __init__(self, dim, out_dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.expand = nn.Linear(dim, 16 * out_dim, bias=False)
        self.norm = norm_layer(out_dim)

    def forward(self, x, H, W):
        x = self.expand(x)
        x = rearrange(x, 'b (h w) (c p1 p2) -> b (h p1) (w p2) c', h=H, w=W, p1=4, p2=4)
        x = rearrange(x, 'b h w c -> b (h w) c')
        x = self.norm(x)
        return x, H * 4, W * 4


class SwinUnet(nn.Module):

    def __init__(self, in_chans=3, num_classes=3, embed_dim=96):
        super().__init__()

        # 1. Patch Embedding (3 -> 96)
        self.patch_embed = PatchEmbed(patch_size=4, in_chans=in_chans, embed_dim=embed_dim)

        # 2. Encoder
        self.layer1 = BasicLayer(dim=embed_dim, depth=2, num_heads=3, window_size=7)
        self.down1 = PatchMerging(dim=embed_dim)

        self.layer2 = BasicLayer(dim=2 * embed_dim, depth=2, num_heads=6, window_size=7)
        self.down2 = PatchMerging(dim=2 * embed_dim)

        self.layer3 = BasicLayer(dim=4 * embed_dim, depth=2, num_heads=12, window_size=7)
        self.down3 = PatchMerging(dim=4 * embed_dim)

        # 3. Bottleneck
        self.layer4 = BasicLayer(dim=8 * embed_dim, depth=2, num_heads=24, window_size=7)

        # 4. Decoder (带 Skip Connection 的纯 Transformer 融合)
        self.up1 = StandardPatchExpanding(input_dim=8 * embed_dim)
        self.concat_linear1 = nn.Linear(8 * embed_dim, 4 * embed_dim)
        self.layer_up1 = BasicLayer(dim=4 * embed_dim, depth=2, num_heads=12, window_size=7)

        self.up2 = StandardPatchExpanding(input_dim=4 * embed_dim)
        self.concat_linear2 = nn.Linear(4 * embed_dim, 2 * embed_dim)
        self.layer_up2 = BasicLayer(dim=2 * embed_dim, depth=2, num_heads=6, window_size=7)

        self.up3 = StandardPatchExpanding(input_dim=2 * embed_dim)
        self.concat_linear3 = nn.Linear(2 * embed_dim, embed_dim)
        self.layer_up3 = BasicLayer(dim=embed_dim, depth=2, num_heads=3, window_size=7)

        # 5. Output
        self.final_up = FinalPatchExpanding(dim=embed_dim, out_dim=embed_dim)
        self.output = nn.Conv2d(embed_dim, num_classes, kernel_size=1)

    def forward(self, x):
        # ---- Encoder ----
        x0, H, W = self.patch_embed(x)  # [B, L, 96]
        x1, H1, W1 = self.layer1(x0, H, W)  # [B, L, 96]

        x, H2, W2 = self.down1(x1, H1, W1)  # [B, L/4, 192]
        x2, H2, W2 = self.layer2(x, H2, W2)  # [B, L/4, 192]

        x, H3, W3 = self.down2(x2, H2, W2)  # [B, L/16, 384]
        x3, H3, W3 = self.layer3(x, H3, W3)  # [B, L/16, 384]

        x, H4, W4 = self.down3(x3, H3, W3)  # [B, L/64, 768]

        # ---- Bottleneck ----
        x4, H4, W4 = self.layer4(x, H4, W4)  # [B, L/64, 768]

        # ---- Decoder ----
        # Up 1
        y, Hu, Wu = self.up1(x4, H4, W4)
        y = torch.cat([y, x3], dim=-1)
        y = self.concat_linear1(y)
        y, Hu, Wu = self.layer_up1(y, Hu, Wu)

        # Up 2
        y, Hu, Wu = self.up2(y, Hu, Wu)
        y = torch.cat([y, x2], dim=-1)
        y = self.concat_linear2(y)
        y, Hu, Wu = self.layer_up2(y, Hu, Wu)

        # Up 3
        y, Hu, Wu = self.up3(y, Hu, Wu)
        y = torch.cat([y, x1], dim=-1)
        y = self.concat_linear3(y)
        y, Hu, Wu = self.layer_up3(y, Hu, Wu)

        # ---- Output ----
        y, Hu, Wu = self.final_up(y, Hu, Wu)
        y = rearrange(y, 'b (h w) c -> b c h w', h=Hu, w=Wu)
        out = self.output(y)

        return out


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 标准 RGB 图像输入: Batch Size=1, Channels=3, H=224, W=224 (或 256)
    img = torch.rand((1, 3, 512, 512)).to(device)

    swin_unet = SwinUnet(in_chans=3, num_classes=3, embed_dim=96).to(device)

    out = swin_unet(img)
    print(f"输入形状: {img.shape}")
    print(f"输出形状: {out.shape}")