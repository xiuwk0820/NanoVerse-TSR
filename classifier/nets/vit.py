from collections import OrderedDict

import torch
from torch import nn


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)

# 原始方案
class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        #-----------------------------------------------#
        #   224, 224, 3 -> 196, 768
        #-----------------------------------------------#
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        #--------------------------------------------------------------------------------------------------------------------#
        #   class_embedding部分是transformer的分类特征。用于堆叠到序列化后的图片特征中，作为一个单位的序列特征进行特征提取。
        #
        #   在利用步长为16x16的卷积将输入图片划分成14x14的部分后，将14x14部分的特征平铺，一幅图片会存在序列长度为196的特征。
        #   此时生成一个class_embedding，将class_embedding堆叠到序列长度为196的特征上，获得一个序列长度为197的特征。
        #   在特征提取的过程中，class_embedding会与图片特征进行特征的交互。最终分类时，我们取出class_embedding的特征，利用全连接分类。
        #--------------------------------------------------------------------------------------------------------------------#
        #   196, 768 -> 197, 768
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        #--------------------------------------------------------------------------------------------------------------------#
        #   为网络提取到的特征添加上位置信息。
        #   以输入图片为224, 224, 3为例，我们获得的序列化后的图片特征为196, 768。加上class_embedding后就是197, 768
        #   此时生成的pos_Embedding的shape也为197, 768，代表每一个特征的位置信息。
        #--------------------------------------------------------------------------------------------------------------------#
        #   197, 768 -> 197, 768
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x

# 方案2，简单化的网络
class SimpleVisionEncoder(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.patch_size = patch_size
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)
        # 阶段1：分块嵌入（替代原VisionTransformer的conv1）
        self.patch_embed = nn.Sequential(
            nn.Conv2d(3, width//4, kernel_size=7, stride=2, padding=3),  # 下采样
            nn.GELU(),
            nn.Conv2d(width//4, width, kernel_size=patch_size, stride=patch_size)  # 分块投影
        )
        
        # 阶段2：轻量级特征提取（替代Transformer）
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1),
            nn.LayerNorm([width, (input_resolution//patch_size)//2, (input_resolution//patch_size)//2]),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(width, width, 3, padding=1),
            nn.LayerNorm([width, (input_resolution//patch_size)//4, (input_resolution//patch_size)//4]),
            nn.GELU()
        )
        
        # 阶段3：全局特征生成（替代class_embedding）
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(width, output_dim)
    
    def get_device(self) -> torch.device:
        """兼容单卡与DataParallel的设备获取方法"""
        try:
            # 通过第一个卷积层的参数获取设备
            return next(self.patch_embed[0].parameters()).device
        except StopIteration:
            # 防御性编程：若无参数默认返回主设备
            return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    def forward(self, x):
        # 分块嵌入 [B,3,224,224] -> [B,width,14,14]（当patch_size=16时）
        x = self.patch_embed(x)
        
        # 特征提取 [B,width,14,14] -> [B,width,3,3]
        x = self.feature_extractor(x)
        
        # 全局特征 [B,width,3,3] -> [B,width]
        x = self.global_pool(x).squeeze(-1).squeeze(-1)
        
        # 投影到输出维度 [B,width] -> [B,output_dim]
        return self.proj(x)


# Mamba Out
class GatedCNNBlock(nn.Module):
    """基于网页3的门控卷积块设计，移除SSM保留核心特征提取能力"""
    def __init__(self, dim, kernel_size=7, expansion_ratio=8/3):
        super().__init__()
        hidden_dim = int(dim * expansion_ratio)
        
        # 核心组件（网页3图1(a)结构）
        self.norm = LayerNorm(dim, eps=1e-6)
        self.dw_conv = nn.Conv2d(dim, dim, kernel_size, 
                               padding=kernel_size//2, 
                               groups=dim)  # 深度可分离卷积
        self.gate_proj = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x):
        # 输入形状: [B, C, H, W]
        shortcut = x
        
        # 特征重整化（网页3步骤1）
        x = self.norm(x.permute(0,2,3,1)).permute(0,3,1,2)
        
        # 多尺度特征混合（网页2的深度卷积优化）
        spatial_feat = self.dw_conv(x)  # [B,C,H,W]
        
        # 门控机制（网页3步骤3）
        gate = self.gate_proj(
            x.permute(0,2,3,1).mean(dim=(1,2))  # 全局平均池化
        ).sigmoid().unsqueeze(-1).unsqueeze(-1)
        
        # 特征融合（网页3步骤4）
        x = spatial_feat * gate + self.mlp(
            x.permute(0,2,3,1)
        ).permute(0,3,1,2)
        
        return x + shortcut  # 残差连接（网页3步骤5）

def build_mambaout_extractor(width, resolution):
    return nn.Sequential(
        # Stage 1 [14x14 -> 7x7]
        nn.Sequential(
            GatedCNNBlock(width),
            GatedCNNBlock(width),
            nn.Conv2d(width, width, 3, stride=2, padding=1)  # 下采样
        ),
        # Stage 2 [7x7 -> 3x3]
        nn.Sequential(
            GatedCNNBlock(width),
            GatedCNNBlock(width),
            nn.AdaptiveAvgPool2d((3,3))  # 替代固定下采样
        ),
        # 通道增强
        nn.Conv2d(width, width*2, 1),
        nn.GELU(),
        nn.Conv2d(width*2, width, 1)
    )

class OutVisionEncoder(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.patch_size = patch_size
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)
        # 阶段1：分块嵌入（替代原VisionTransformer的conv1）
        self.patch_embed = nn.Sequential(
            nn.Conv2d(3, width//4, kernel_size=7, stride=2, padding=3),  # 下采样
            nn.GELU(),
            nn.Conv2d(width//4, width, kernel_size=patch_size, stride=patch_size)  # 分块投影
        )
        
        # 阶段2：轻量级特征提取（替代Transformer）
        self.feature_extractor = build_mambaout_extractor(width, input_resolution//patch_size)
        # 阶段3：全局特征生成（替代class_embedding）
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(width, output_dim)
    
    def get_device(self) -> torch.device:
        """兼容单卡与DataParallel的设备获取方法"""
        try:
            # 通过第一个卷积层的参数获取设备
            return next(self.patch_embed[0].parameters()).device
        except StopIteration:
            # 防御性编程：若无参数默认返回主设备
            return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    def forward(self, x):
        # 分块嵌入 [B,3,224,224] -> [B,width,14,14]（当patch_size=16时）
        x = self.patch_embed(x)

        # 特征提取 [B,width,14,14] -> [B,width,3,3]
        x = self.feature_extractor(x)
        
        # 全局特征 [B,width,3,3] -> [B,width]
        x = self.global_pool(x).squeeze(-1).squeeze(-1)
        
        # 投影到输出维度 [B,width] -> [B,output_dim]
        return self.proj(x)

