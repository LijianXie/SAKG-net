"""
SAKG.py - Optimized Version（带空间降维以防止OOM）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

class SelfAttention(nn.Module):
    """全局时空注意力机制（带空间降维以防止OOM）"""
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
        # 降维池化：将K和V的空间尺度固定在16x16，大幅降低N^2复杂度
        self.pool = nn.AdaptiveAvgPool2d((16, 16))

    def forward(self, x):
        B, C, H, W = x.size()
        Q = self.query(x).view(B, -1, H * W)

        x_pooled = self.pool(x)
        K = self.key(x_pooled).view(B, -1, 256)
        V = self.value(x_pooled).view(B, -1, 256)

        # 带有 d_k 缩放因子的点积注意力
        attention = torch.bmm(Q.transpose(1, 2), K) / math.sqrt(C // 8)
        attention = self.softmax(attention)

        out = torch.bmm(V, attention.transpose(1, 2)).view(B, C, H, W)
        return self.gamma * out + x

class DynamicGraphConvolution(nn.Module):
    """基于拉普拉斯矩阵归一化的图卷积"""
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
    """聚类与图卷积块（可微分软分配机制）"""
    def __init__(self, channels, max_k=16):
        super(ClusterGCNBlock, self).__init__()
        self.max_k = max_k
        self.cluster_proj = nn.Conv2d(channels, max_k, kernel_size=1)
        self.gcn = DynamicGraphConvolution(channels)
        self.out_proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.size()

        # 1. 自适应 K 计算 (基于方差的模拟)
        var = torch.var(x).item()
        K = int(max(4, min(self.max_k, var * 10)))

        # 2. 可微分软分配聚类 (Soft-Assignment)
        assign_logits = self.cluster_proj(x)[:, :K, :, :] # 取出前 K 个聚类头
        mask = F.softmax(assign_logits.view(B, K, H * W), dim=1) # [B, K, N]

        # 3. 节点池化
        x_flat = x.view(B, C, H * W).transpose(1, 2)
        node_feat = torch.bmm(mask, x_flat) # [B, K, C]

        # 4. 图卷积
        node_feat_enhanced = self.gcn(node_feat)

        # 5. 反投影回像素空间
        out_flat = torch.bmm(mask.transpose(1, 2), node_feat_enhanced)
        out = out_flat.transpose(1, 2).view(B, C, H, W)
        return self.out_proj(out)

class AdaptiveFusion(nn.Module):
    """自适应特征融合: F_fused = W1 * F_c + W2 * F_g"""
    def __init__(self, channels):
        super(AdaptiveFusion, self).__init__()
        self.w_conv = nn.Conv2d(channels * 2, 2, kernel_size=1)

    def forward(self, f_c, f_g):
        weights = F.softmax(self.w_conv(torch.cat([f_c, f_g], dim=1)), dim=1)
        return f_c * weights[:, 0:1] + f_g * weights[:, 1:2]

class EncoderStage(nn.Module):
    """完整编码阶段：Conv -> GCN -> Fuse -> Attention"""
    def __init__(self, in_c, out_c, stride=1):
        super(EncoderStage, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
        self.gcn = ClusterGCNBlock(out_c)
        self.fusion = AdaptiveFusion(out_c)
        self.attn = SelfAttention(out_c)

    def forward(self, x):
        f_c = self.conv(x)          # CNN 分支提取局部特征
        f_g = self.gcn(f_c)         # GCN 分支提取图结构特征
        f_fused = self.fusion(f_c, f_g) # 自适应融合
        out = self.attn(f_fused)    # 全局注意力增强
        return out

class SAKG_Pretrain(nn.Module):
    def __init__(self, in_channels=3):
        super(SAKG_Pretrain, self).__init__()
        # 4 层卷积 (4 conv layers)
        self.stage1 = EncoderStage(in_channels, 64, stride=2)   # 256 -> 128
        self.stage2 = EncoderStage(64, 128, stride=2)           # 128 -> 64
        self.stage3 = EncoderStage(128, 256, stride=2)          # 64 -> 32
        self.stage4 = EncoderStage(256, 512, stride=2)          # 32 -> 16

        # 对比学习投影头
        self.projector = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128)
        )

    def forward_features(self, x):
        e1 = self.stage1(x)
        e2 = self.stage2(e1)
        e3 = self.stage3(e2)
        e4 = self.stage4(e3)
        return e4

    def forward(self, x_a, x_b):
        z_a = self.projector(self.forward_features(x_a))
        z_b = self.projector(self.forward_features(x_b))
        return z_a, z_b

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing FULL SAKG Pretrain on: {device}")
    model = SAKG_Pretrain(in_channels=3).to(device)
    x_a = torch.randn(2, 3, 256, 256).to(device)
    x_b = torch.randn(2, 3, 256, 256).to(device)
    z_a, z_b = model(x_a, x_b)
    print(f"Output Z_a shape: {z_a.shape}!")