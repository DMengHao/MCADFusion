import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops import rearrange


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + \
                    torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class AttentionBase(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False, ):
        super(AttentionBase, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv1 = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=qkv_bias)
        self.qkv2 = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, padding=1, bias=qkv_bias)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv2(self.qkv1(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)
        out = self.proj(out)
        return out


class Mlp(nn.Module):

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 ffn_expansion_factor=2,
                 bias=False):
        super().__init__()
        hidden_features = int(in_features * ffn_expansion_factor)

        self.project_in = nn.Conv2d(
            in_features, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features, bias=bias)

        self.project_out = nn.Conv2d(
            hidden_features, in_features, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x
class BaseFeatureExtraction(nn.Module):
    def __init__(self, dim, num_heads):
        super(BaseFeatureExtraction, self).__init__()
        # self.block1 = nn.Sequential(
        #     Inception(64, 64, (96, 128), (16, 32), 32),
        # )
        # self.block2 = nn.Sequential(
        #     nn.Conv2d(in_channels=256, out_channels=64, kernel_size=1, stride=1, padding=0)
        # )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # 利用1x1卷积代替全连接
        self.fc1 = nn.Conv2d(in_channels=dim, out_channels=dim // num_heads, kernel_size=1, bias=False)
        # 效果不是老么好
        # self.bn    =nn.BatchNorm2d(dim//num_heads)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels=dim // num_heads, out_channels=dim, kernel_size=1, bias=False)
        # self.block1 = nn.Sequential(
        #     Inception(dim, 64, (96, 128), (16, 32), 32),
        #     nn.Conv2d(in_channels=256, out_channels=64, kernel_size=1, stride=1, padding=0)
        # )
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        # y=self.block2(self.block1(x))
        y = x
        # # y=self.block1(y)
        # avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        # max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        # out = avg_out + max_out
        # # out = avg_out
        return y
class InvertedResidualBlock(nn.Module):
    def __init__(self, inp, oup, expand_ratio):
        super(InvertedResidualBlock, self).__init__()
        hidden_dim = int(inp * expand_ratio)
        self.bottleneckBlock = nn.Sequential(
            # pw
            nn.Conv2d(inp, hidden_dim, 1, bias=False),
            # nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # dw
            nn.ReflectionPad2d(1),
            nn.Conv2d(hidden_dim, hidden_dim, 3, groups=hidden_dim, bias=False),
            # nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, bias=False),
            # nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        return self.bottleneckBlock(x)

class DetailNode(nn.Module):
    def __init__(self):
        super(DetailNode, self).__init__()
        # Scale is Ax + b, i.e. affine transformation
        self.theta_phi = InvertedResidualBlock(inp=32, oup=32, expand_ratio=2)
        self.theta_rho = InvertedResidualBlock(inp=32, oup=32, expand_ratio=2)
        self.theta_eta = InvertedResidualBlock(inp=32, oup=32, expand_ratio=2)
        self.shffleconv = nn.Conv2d(64, 64, kernel_size=1,
                                    stride=1, padding=0, bias=True)

    def separateFeature(self, x):
        z1, z2 = x[:, :x.shape[1] // 2], x[:, x.shape[1] // 2:x.shape[1]]
        return z1, z2

    def forward(self, z1, z2):
        z1, z2 = self.separateFeature(
            self.shffleconv(torch.cat((z1, z2), dim=1)))
        z2 = z2 + self.theta_phi(z1)
        z1 = z1 * torch.exp(self.theta_rho(z2)) + self.theta_eta(z2)
        return z1, z2


# 空间注意力机制替换细节提取
# SpatialAttention——>
class DetailFeatureExtraction(nn.Module):
    def __init__(self, kernel_size=7):
        super(DetailFeatureExtraction, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = x
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        x = self.sigmoid(x)
        x = x * y
        return x



import numbers

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')
def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape
    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight
class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape
    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias
class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)
    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)
# 前馈神经网络 其作用是对输入特征进行非线性变换
class FeedForward(nn.Module):
    # 64 2 False
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        #  64 256 1 False
        self.project_in = nn.Conv2d(
            dim, hidden_features * 2, kernel_size=1, bias=bias)
        # 256 256 3 1 1 256 False
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features * 2, bias=bias)
        # 128 64 1 False
        self.project_out = nn.Conv2d(
            hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        # (8 64 128 128)->(8 256 128 128)
        x = self.project_in(x)
        # (8 256 128 128)->(8 256 128 128)
        # (8 128 128 128) (8 128 128 128)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        # 经过GELU激活函数之后 在相乘(8 128 128 128)
        x = F.gelu(x1) * x2
        # (8 128 128 128)->(8 64 128 128)
        x = self.project_out(x)
        return x

# Attention(Q,K,V) 输出是经过自注意力加权后的特征张量，具有与输入相同的维度和形状。
class Attention(nn.Module):
    # 自注意力模块的作用是对输入特征图进行自注意力计算，从而获取每个位置的重要程度，并生成对应的输出特征。
    # 64 8 Flase
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        # 8行一列 全1 温度参数来调节注意力的尖峰度
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        # 64 64*3 1 Flase
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        # 64*3 64*3 3 1 1 64*3 False
        # groups=64*3 每个输入通道连接对应的输出通道不会连接到其他的输出通道
        # 用来捕获特征的局部相关性
        self.qkv_dwconv = nn.Conv2d(
            dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        # 64 64 1 False
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        # (8 64 128 128)
        b, c, h, w = x.shape
        # (8 64 128 128) qkv(x) ->(8 192 128 128)
        # (8 192 128 128) qkv_dwconv ->(8 192 128 128)
        qkv = self.qkv_dwconv(self.qkv(x))
        # 64 64 64
        q, k, v = qkv.chunk(3, dim=1)
        # (8 (8 192) 128 128)->(8 8 192 (128 128))
        q = rearrange(q, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        # 沿着最后一个维度进行归一化 映射到（0,1）
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        # Q.K*temperature
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        # 注意力分数进行 softmax 归一化操作，使得每个位置的注意力权重在该行上总和为 1。
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        # (8 8 192 (128 128))->(8 (8 192) 128 128)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)
        # Cov2d(64,64,1,False)
        out = self.project_out(out)
        return out

class TransformerBlock(nn.Module):
    # 64 8 2 False WithBias
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()
        # 每个样本的特征进行均值为 0、方差为 1 的标准化处理
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        # 64 8 False 输出是经过自注意力加权后的特征张量，具有与输入相同的维度和形状。
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        # 64 2 False 前馈神经网络其作用是对输入特征进行非线性变换
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        # Attention(Q,K,V) 输出是经过自注意力加权后的特征张量，具有与输入相同的维度和形状。
        x = x + self.attn(self.norm1(x))
        #  前馈神经网络 其作用是对输入特征进行非线性变换
        x = x + self.ffn(self.norm2(x))
        return x

class Inception(nn.Module):
    def __init__(self, in_channels, c1, c2, c3, c4):
        super(Inception, self).__init__()
        self.ReLU = nn.ReLU()
        # 路线1，单1×1卷积层
        self.p1_1 = nn.Conv2d(in_channels=in_channels, out_channels=c1, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(c1)
        # 路线2，1×1卷积层, 3×3的卷积
        self.p2_1 = nn.Conv2d(in_channels=in_channels, out_channels=c2[0], kernel_size=1)
        self.bn2 = nn.BatchNorm2d(c2[0])
        self.p2_2 = nn.Conv2d(in_channels=c2[0], out_channels=c2[1], kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(c2[1])
        # 路线3，1×1卷积层, 5×5的卷积
        self.p3_1 = nn.Conv2d(in_channels=in_channels, out_channels=c3[0], kernel_size=1)
        self.bn4 = nn.BatchNorm2d(c3[0])
        self.p3_2 = nn.Conv2d(in_channels=c3[0], out_channels=c3[1], kernel_size=5, padding=2)
        self.bn5 = nn.BatchNorm2d(c3[1])
        # 路线4，3×3的最大池化, 1×1的卷积
        self.p4_1 = nn.MaxPool2d(kernel_size=3, padding=1, stride=1)
        self.p4_2 = nn.Conv2d(in_channels=in_channels, out_channels=c4, kernel_size=1)
        self.bn6 = nn.BatchNorm2d(c4)
    def forward(self, x):
        # p1 = self.ReLU(self.p1_1(x))
        # p2 = self.ReLU(self.p2_2(self.ReLU(self.p2_1(x))))
        # p3 = self.ReLU(self.p3_2(self.ReLU(self.p3_1(x))))
        # p4 = self.ReLU(self.p4_2(self.p4_1(x)))
        p1 = self.ReLU(self.bn1(self.p1_1(x)))
        p2 = self.ReLU(self.bn3(self.p2_2(self.ReLU(self.bn2(self.p2_1(x))))))
        p3 = self.ReLU(self.bn5(self.p3_2(self.ReLU(self.bn4(self.p3_1(x))))))
        p4 = self.ReLU(self.bn6(self.p4_2(self.p4_1(x))))
        return torch.cat((p1, p2, p3, p4), dim=1)

class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=64, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3,
                              stride=1, padding=1, bias=bias)
        self.block1 = nn.Sequential(
            Inception(embed_dim, 64, (96, 128), (16, 32), 32),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        # x=self.proj(x)
        # x = self.block1(self.proj(x))
        # x = self.block2(x)
        x = self.block1(self.proj(x))
        x = self.block2(x)
        return x

# class OverlapPatchEmbed2(nn.Module):
#     def __init__(self, in_c=64, embed_dim=64, bias=False):
#         super(OverlapPatchEmbed2, self).__init__()
#
#
#         self.block1 = nn.Sequential(
#             Inception(embed_dim, 64,(96,128),(16,32),32),
#         )
#         self.block2 = nn.Sequential(
#             nn.Conv2d(in_channels=256,out_channels=64,kernel_size=1,stride=1,padding=0)
#         )
#     def forward(self, x):
#         x = self.block1(x)
#         x = self.block2(x)
#         return x

class Restormer_Encoder(nn.Module):
    def __init__(self,
                 inp_channels=1,
                 out_channels=1,
                 dim=64,
                 num_blocks=[4, 4],
                 heads=[8, 8, 8],
                 ffn_expansion_factor=2,
                 bias=False,
                 LayerNorm_type='WithBias',
                 ):
        super(Restormer_Encoder, self).__init__()
        # 对输入图片进行处理 不改变大小 改变通道数为64 (8 1 128 128)->(8 64 128 128)
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        # self.patch_embed2 = OverlapPatchEmbed2(inp_channels, dim)
        # self.patch_embed = ResNet18()
        self.encoder_level1 = nn.Sequential(
            *[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                               bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.baseFeature = BaseFeatureExtraction(dim=dim, num_heads=16)
        self.detailFeature = DetailFeatureExtraction()
    def forward(self, inp_img):
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        # out_enc_level1 = self.patch_embed2(out_enc_level1)
        base_feature = self.baseFeature(out_enc_level1)
        detail_feature = self.detailFeature(out_enc_level1)
        return base_feature, detail_feature, out_enc_level1

class Restormer_Decoder(nn.Module):
    def __init__(self,
                 # 输入通道数
                 inp_channels=1,
                 # 输出通道数
                 out_channels=1,
                 dim=64,

                 num_blocks=[4, 4],
                 heads=[8, 8, 8],
                 ffn_expansion_factor=2,
                 bias=False,
                 LayerNorm_type='WithBias',
                 ):
        super(Restormer_Decoder, self).__init__()
        self.reduce_channel = nn.Conv2d(int(dim * 2), int(dim), kernel_size=1, bias=bias)
        self.encoder_level2 = nn.Sequential(
            *[TransformerBlock(dim=dim, num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                               bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        self.output = nn.Sequential(
            nn.Conv2d(int(dim), int(dim) // 2, kernel_size=3,
                      stride=1, padding=1, bias=bias),
            nn.LeakyReLU(),
            nn.Conv2d(int(dim) // 2, out_channels, kernel_size=3,
                      stride=1, padding=1, bias=bias), )
        self.sigmoid = nn.Sigmoid()
    def forward(self, inp_img, base_feature, detail_feature):
        out_enc_level0 = torch.cat((base_feature, detail_feature), dim=1)
        out_enc_level0 = self.reduce_channel(out_enc_level0)
        out_enc_level1 = self.encoder_level2(out_enc_level0)
        # 使用残差结构
        if inp_img is not None:
            out_enc_level1 = self.output(out_enc_level1) + inp_img
        else:
            out_enc_level1 = self.output(out_enc_level1)
        return self.sigmoid(out_enc_level1), out_enc_level0