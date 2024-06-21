# （1）CBAM注意力机制
# 通道注意力
import torch
import tensorflow
from tensorflow.keras import layers
from torch import nn


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 利用1x1卷积代替全连接
        self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


# 空间注意力机制
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


# CBAM注意力
def CBAM_attention(inputs):
    # 先经过通道注意力再经过空间注意力
    x = ChannelAttention(inputs)
    x = SpatialAttention(x)
    return x


# （2）# 深度可分离卷积+BN+relu
def conv_block(inputs, filters, kernel_size, strides):
    # 深度卷积
    x = layers.DepthwiseConv2D(kernel_size, strides,
                               padding='same', use_bias=False)(inputs)  # 有BN不要偏置
    # 逐点卷积
    x = layers.Conv2D(filters, kernel_size=(1, 1), strides=1,
                      padding='same', use_bias=False)(x)

    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    return x
# （3）5次卷积操作提取特征减少参数量
def five_conv(x, filters):
    x = conv_block(x, filters, (1, 1), strides=1)
    x = conv_block(x, filters * 2, (3, 3), strides=1)
    x = conv_block(x, filters, (1, 1), strides=1)
    x = conv_block(x, filters * 2, (3, 3), strides=1)
    x = conv_block(x, filters, (1, 1), strides=1)
    return x


# （4）主干网络
def panet(feat1, feat2, p5):
    # 对网络的有效输出特征层采用CBAM注意力
    feat1 = CBAM_attention(feat1)
    feat2 = CBAM_attention(feat2)
    p5 = CBAM_attention(p5)

    # （1）
    # 对spp结构的输出进行卷积和上采样
    # [13,13,512]==>[13,13,256]==>[26,26,256]
    p5_upsample = conv_block(p5, filters=256, kernel_size=(1, 1), strides=1)
    p5_upsample = layers.UpSampling2D(size=(2, 2))(p5_upsample)

    # 对feat2特征层卷积后再与p5_upsample堆叠
    # [26,26,512]==>[26,26,256]==>[26,26,512]
    p4 = conv_block(feat2, filters=256, kernel_size=(1, 1), strides=1)
    # 上采样后使用注意力机制
    p4 = CBAM_attention(p4)
    # 通道维度上堆叠
    p4 = layers.concatenate([p4, p5_upsample])

    # 堆叠后进行5次卷积[26,26,512]==>[26,26,256]
    p4 = five_conv(p4, filters=256)

    # （2）
    # 对p4卷积上采样
    # [26,26,256]==>[26,26,512]==>[52,52,512]
    p4_upsample = conv_block(p4, filters=128, kernel_size=(1, 1), strides=1)
    p4_upsample = layers.UpSampling2D(size=(2, 2))(p4_upsample)

    # feat1层卷积后与p4_upsample堆叠
    # [52,52,256]==>[52,52,128]==>[52,52,256]
    p3 = conv_block(feat1, filters=128, kernel_size=(1, 1), strides=1)
    # 上采样后使用注意力机制
    p3 = CBAM_attention(p3)
    # 通道维度上堆叠
    p3 = layers.concatenate([p3, p4_upsample])

    # 堆叠后进行5次卷积[52,52,256]==>[52,52,128]
    p3 = five_conv(p3, filters=128)

    # 存放第一个特征层的输出
    p3_output = p3

    # （3）
    # p3卷积下采样和p4堆叠
    # [52,52,128]==>[26,26,256]==>[26,26,512]
    p3_downsample = conv_block(p3, filters=256, kernel_size=(3, 3), strides=2)
    # 下采样后使用注意力机制
    p3_downsample = CBAM_attention(p3_downsample)
    # 通道维度上堆叠
    p4 = layers.concatenate([p3_downsample, p4])

    # 堆叠后的结果进行5次卷积[26,26,512]==>[26,26,256]
    p4 = five_conv(p4, filters=256)

    # 存放第二个有效特征层的输出
    p4_output = p4

    # （4）
    # p4卷积下采样和p5堆叠
    # [26,26,256]==>[13,13,512]==>[13,13,1024]
    p4_downsample = conv_block(p4, filters=512, kernel_size=(3, 3), strides=2)
    # 下采样后使用注意力机制
    p4_downsample = CBAM_attention(p4_downsample)
    # 通道维度上堆叠
    p5 = layers.concatenate([p4_downsample, p5])

    # 堆叠后进行5次卷积[13,13,1024]==>[13,13,512]
    p5 = five_conv(p5, filters=512)

    # 存放第三个有效特征层的输出
    p5_output = p5

    # 返回输出层结果
    return p3_output, p4_output, p5_output