"""
AKG.py - Optimized Version（带空间降维以防止OOM）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


# ----------------- 核心基础模块 (与 SAKG 完全一致) -----------------
class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
        self.pool = nn.AdaptiveAvgPool2d((16, 16))

    def forward(self, x):
        B, C, H, W = x.size()
        Q = self.query(x).view(B, -1, H * W)
        x_pooled = self.pool(x)
        K = self.key(x_pooled).view(B, -1, 256)
        V = self.value(x_pooled).view(B, -1, 256)
        attention = torch.bmm(Q.transpose(1, 2), K) / math.sqrt(C // 8)
        attention = self.softmax(attention)
        out = torch.bmm(V, attention.transpose(1, 2)).view(B, C, H, W)
        return self.gamma * out + x


class DynamicGraphConvolution(nn.Module):
    def __init__(self, in_channels):
        super(DynamicGraphConvolution, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_channels, in_channels))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, node_feat):
        A = F.relu(torch.bmm(node_feat, node_feat.transpose(1, 2)))
        D_inv = torch.pow(A.sum(-1, keepdim=True) + 1e-8, -0.5)
        A_norm = A * D_inv * D_inv.transpose(1, 2)
        out = torch.matmul(torch.bmm(A_norm, node_feat), self.weight)
        return F.relu(out)


class ClusterGCNBlock(nn.Module):
    def __init__(self, channels, max_k=16):
        super(ClusterGCNBlock, self).__init__()
        self.cluster_proj = nn.Conv2d(channels, max_k, kernel_size=1)
        self.gcn = DynamicGraphConvolution(channels)
        self.out_proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.size()
        K = int(max(4, min(16, torch.var(x).item() * 10)))
        mask = F.softmax(self.cluster_proj(x)[:, :K, :, :].view(B, K, H * W), dim=1)
        node_feat = torch.bmm(mask, x.view(B, C, H * W).transpose(1, 2))
        node_feat_enhanced = self.gcn(node_feat)
        out = torch.bmm(mask.transpose(1, 2), node_feat_enhanced).transpose(1, 2).view(B, C, H, W)
        return self.out_proj(out)


class AdaptiveFusion(nn.Module):
    def __init__(self, channels):
        super(AdaptiveFusion, self).__init__()
        self.w_conv = nn.Conv2d(channels * 2, 2, kernel_size=1)

    def forward(self, f_c, f_g):
        weights = F.softmax(self.w_conv(torch.cat([f_c, f_g], dim=1)), dim=1)
        return f_c * weights[:, 0:1] + f_g * weights[:, 1:2]


class EncoderStage(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super(EncoderStage, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
        self.gcn = ClusterGCNBlock(out_c)
        self.fusion = AdaptiveFusion(out_c)
        self.attn = SelfAttention(out_c)

    def forward(self, x):
        f_c = self.conv(x)
        f_g = self.gcn(f_c)
        return self.attn(self.fusion(f_c, f_g))


# ----------------- 解码器特定模块 -----------------
class DecoderStage(nn.Module):
    """带跳跃连接的解码层"""

    def __init__(self, in_c, skip_c, out_c):
        super(DecoderStage, self).__init__()
        # 上采样将通道减半
        self.up = nn.ConvTranspose2d(in_c, in_c // 2, kernel_size=2, stride=2)
        # 拼接后通道为 (in_c // 2 + skip_c)
        self.conv = nn.Sequential(
            nn.Conv2d(in_c // 2 + skip_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
        self.attn = SelfAttention(out_c)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            # 保证尺寸匹配后进行拼接
            x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return self.attn(x)


class AKG_Supervised(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super(AKG_Supervised, self).__init__()

        # 4 层 Encoder
        self.enc1 = EncoderStage(in_channels, 64, stride=2)  # 256 -> 128
        self.enc2 = EncoderStage(64, 128, stride=2)  # 128 -> 64
        self.enc3 = EncoderStage(128, 256, stride=2)  # 64 -> 32
        self.enc4 = EncoderStage(256, 512, stride=2)  # 32 -> 16

        # 4 层 Decoder
        self.dec4 = DecoderStage(in_c=512, skip_c=256, out_c=256)  # up(16)->32, cat e3(256)
        self.dec3 = DecoderStage(in_c=256, skip_c=128, out_c=128)  # up(32)->64, cat e2(128)
        self.dec2 = DecoderStage(in_c=128, skip_c=64, out_c=64)  # up(64)->128, cat e1(64)

        # 最后一层：恢复到原始分辨率 256x256 (不使用 skip 连接，或使用最原始图像特征)
        self.dec1 = DecoderStage(in_c=64, skip_c=0, out_c=64)  # up(128)->256

        # 输出预测掩码
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # 编码阶段
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        # 解码阶段 (带着跳跃连接)
        d4 = self.dec4(e4, skip=e3)
        d3 = self.dec3(d4, skip=e2)
        d2 = self.dec2(d3, skip=e1)
        d1 = self.dec1(d2, skip=None)  # 最后一层恢复原始尺寸

        out = self.final_conv(d1)
        return out


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing FULL AKG Supervised on: {device}")

    num_classes = 10
    model = AKG_Supervised(in_channels=3, num_classes=num_classes).to(device)

    img = torch.randn(1, 3, 256, 256).to(device)
    output = model(img)

    print(f"Output Map shape: {output.shape} (Expected: [1, 10, 256, 256])")
