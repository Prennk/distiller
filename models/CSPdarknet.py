import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

class GlobalAvgPool2d(torch.nn.Module):
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()

    @staticmethod
    def forward(x):
        return torch.mean(x.view(x.size(0), x.size(1), -1), dim=2)

#-------------------------------------------------#
#   MISH激活函数
#-------------------------------------------------#
class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

#---------------------------------------------------#
#   卷积块 -> 卷积 + 标准化 + 激活函数
#   Conv2d + BatchNormalization + Mish
#---------------------------------------------------#
class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(BasicConv, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size//2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = Mish()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x

#---------------------------------------------------#
#   CSPdarknet的结构块的组成部分
#   内部堆叠的残差块
#---------------------------------------------------#
class Resblock(nn.Module):
    def __init__(self, channels, hidden_channels=None):
        super(Resblock, self).__init__()

        if hidden_channels is None:
            hidden_channels = channels

        self.block = nn.Sequential(
            BasicConv(channels, hidden_channels, 1),
            BasicConv(hidden_channels, channels, 3)
        )

    def forward(self, x):
        return x + self.block(x)

#--------------------------------------------------------------------#
#   CSPdarknet的结构块
#   首先利用ZeroPadding2D和一个步长为2x2的卷积块进行高和宽的压缩
#   然后建立一个大的残差边shortconv、这个大残差边绕过了很多的残差结构
#   主干部分会对num_blocks进行循环，循环内部是残差结构。
#   对于整个CSPdarknet的结构块，就是一个大残差块+内部多个小残差块
#--------------------------------------------------------------------#
class Resblock_body(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks, first):
        super(Resblock_body, self).__init__()
        #----------------------------------------------------------------#
        #   利用一个步长为2x2的卷积块进行高和宽的压缩
        #----------------------------------------------------------------#
        self.downsample_conv = BasicConv(in_channels, out_channels, 3, stride=2)

        if first:
            #--------------------------------------------------------------------------#
            #   然后建立一个大的残差边self.split_conv0、这个大残差边绕过了很多的残差结构
            #--------------------------------------------------------------------------#
            self.split_conv0 = BasicConv(out_channels, out_channels, 1)

            #----------------------------------------------------------------#
            #   主干部分会对num_blocks进行循环，循环内部是残差结构。
            #----------------------------------------------------------------#
            self.split_conv1 = BasicConv(out_channels, out_channels, 1)  
            self.blocks_conv = nn.Sequential(
                Resblock(channels=out_channels, hidden_channels=out_channels//2),
                BasicConv(out_channels, out_channels, 1)
            )

            self.concat_conv = BasicConv(out_channels*2, out_channels, 1)
        else:
            #--------------------------------------------------------------------------#
            #   然后建立一个大的残差边self.split_conv0、这个大残差边绕过了很多的残差结构
            #--------------------------------------------------------------------------#
            self.split_conv0 = BasicConv(out_channels, out_channels//2, 1)

            #----------------------------------------------------------------#
            #   主干部分会对num_blocks进行循环，循环内部是残差结构。
            #----------------------------------------------------------------#
            self.split_conv1 = BasicConv(out_channels, out_channels//2, 1)
            self.blocks_conv = nn.Sequential(
                *[Resblock(out_channels//2) for _ in range(num_blocks)],
                BasicConv(out_channels//2, out_channels//2, 1)
            )

            self.concat_conv = BasicConv(out_channels, out_channels, 1)

    def forward(self, x):
        x = self.downsample_conv(x)

        x0 = self.split_conv0(x)

        x1 = self.split_conv1(x)
        x1 = self.blocks_conv(x1)

        #------------------------------------#
        #   将大残差边再堆叠回来
        #------------------------------------#
        x = torch.cat([x1, x0], dim=1)
        #------------------------------------#
        #   最后对通道数进行整合
        #------------------------------------#
        x = self.concat_conv(x)

        return x

#---------------------------------------------------#
#   CSPdarknet53 的主体部分
#   输入为一张416x416x3的图片
#   输出为三个有效特征层
#---------------------------------------------------#
class CSPDarkNet(nn.Module):
    def __init__(self, layers, num_classes=1000):
        super(CSPDarkNet, self).__init__()
        self.num_classes =num_classes
        self.inplanes = 32
        # 416,416,3 -> 416,416,32
        self.conv1 = BasicConv(3, self.inplanes, kernel_size=3, stride=1)
        self.feature_channels = [64, 128, 256, 512, 1024]

        self.stages = nn.ModuleList([
            # 416,416,32 -> 208,208,64
            Resblock_body(self.inplanes, self.feature_channels[0], layers[0], first=True),
            # 208,208,64 -> 104,104,128
            Resblock_body(self.feature_channels[0], self.feature_channels[1], layers[1], first=False),
            # 104,104,128 -> 52,52,256
            Resblock_body(self.feature_channels[1], self.feature_channels[2], layers[2], first=False),
            # 52,52,256 -> 26,26,512
            Resblock_body(self.feature_channels[2], self.feature_channels[3], layers[3], first=False),
            # 26,26,512 -> 13,13,1024
            Resblock_body(self.feature_channels[3], self.feature_channels[4], layers[4], first=False)
        ])

        self.num_features = 1
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.classifier = nn.Sequential(
            GlobalAvgPool2d(),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)

        x = self.stages[0](x)
        x = self.stages[1](x)
        out3 = self.stages[2](x)
        out4 = self.stages[3](out3)
        out5 = self.stages[4](out4)
        x = self.classifier(out5)

        return x
    
def darknet53(num_classes=1000, pretrained=False):
    model = CSPDarkNet([1, 2, 8, 8, 4], num_classes=num_classes)
    if pretrained:
        raise ValueError('No Pretrained!!!!')
    return model

# self.backbone.model <--- buat key .model
# class CSPDarkNet53Model(nn.Module):
#     def __init__(self, num_classes=1000, pretrained=False):
#         super(CSPDarkNet53Model, self).__init__()
#         self.model = darknet53(num_classes=num_classes, pretrained=pretrained)

#     def forward(self, x, is_feat=False, preact=False):
#         return self.model(x)

# self.backbone.model <--- buat key .backbone
class CSPDarkNet53Backbone(nn.Module):
    def __init__(self, num_classes=1000):
        super(CSPDarkNet53Backbone, self).__init__()
        self.backbone = darknet53(num_classes=num_classes, pretrained=False)

    def forward(self, x, is_feat=False, preact=False):
        x = self.backbone.conv1(x)
        f0 = x

        x = self.backbone.stages[0](x)
        f1 = x

        x = self.backbone.stages[1](x)
        f2 = x

        x = self.backbone.stages[2](x)
        f3 = x

        x = self.backbone.stages[3](x)
        f4 = x

        x = self.backbone.stages[4](x)
        f5 = x

        x = self.backbone.classifier(x)

        if is_feat:
            return [f0, f1, f2, f3, f4, f5], x
        else:
            return x
        
if __name__ == "__main__":
    # print(mobilenet_v2())
    # from torchinfo import summary
    # model = MobileNetV2(width_mult=0.5)
    # summary(model)

    x = torch.randn(2, 3, 32, 32)
    net = CSPDarkNet53Backbone()
    print(net)
    print()

    feats, logit = net(x, is_feat=True)
    for f in feats:
        print(f.shape, f.min().item())
    print(logit.shape)

    # for m in net.get_bn_before_relu():
    #     if isinstance(m, nn.BatchNorm2d):
    #         print('pass')
    #     else:
    #         print('warning')

