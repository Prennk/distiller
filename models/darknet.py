"""
@author: Yakhyokhuja Valikhujaev <yakhyo9696@gmail.com>
"""
import torch
import torch.nn as nn
from torch.nn import MaxPool2d, functional as F

__all__ = ['darknet19', 'darknet53', 'darknet53e', 'cspdarknet53']

class GlobalAvgPool2d(torch.nn.Module):
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()

    @staticmethod
    def forward(x):
        return torch.mean(x.view(x.size(0), x.size(1), -1), dim=2)


def auto_pad(k, p=None):
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, auto_pad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.LeakyReLU(0.01) if act else nn.ReLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):
    def __init__(self, c1, shortcut=True):
        super(ResidualBlock, self).__init__()
        c2 = c1 // 2
        self.shortcut = shortcut
        self.layer1 = Conv(c1, c2, p=0)
        self.layer2 = Conv(c2, c1, k=3)

    def forward(self, x):
        residual = x
        out = self.layer1(x)
        out = self.layer2(out)
        out += residual
        if self.shortcut:
            out += residual
        return out


class CSP(nn.Module):
    """ [https://arxiv.org/pdf/1911.11929.pdf] """
    def __init__(self, c1, c2, num_blocks=1, shortcut=True, g=1, e=0.5):
        super(CSP, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.conv1 = Conv(c1, c_, 1, 1)
        self.conv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.conv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.conv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.m = nn.Sequential(*[ResidualBlock(c_, shortcut=shortcut) for _ in range(num_blocks)])

    def forward(self, x):
        y1 = self.conv3(self.m(self.conv1(x)))
        y2 = self.conv2(x)
        return self.conv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))


class Elastic(nn.Module):
    """ [https://arxiv.org/abs/1812.05262] """
    def __init__(self, c1):
        super(Elastic, self).__init__()
        c2 = c1 // 2

        self.down = nn.AvgPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self.layer1 = Conv(c1, c2 // 2, p=0)
        self.layer2 = Conv(c2 // 2, c1, k=3)

    def forward(self, x):
        residual = x
        elastic = x

        # check the input size before downsample
        if x.size(2) % 2 > 0 or x.size(3) % 2 > 0:
            elastic = F.pad(elastic, (0, x.size(3) % 2, 0, x.size(2) % 2), mode='replicate')

        down = self.down(elastic)
        elastic = self.layer1(down)
        elastic = self.layer2(elastic)
        up = self.up(elastic)
        # check the output size after upsample
        if up.size(2) > x.size(2) or up.size(3) > x.size(3):
            up = up[:, :, :x.size(2), :x.size(3)]

        half = self.layer1(x)
        half = self.layer2(half)

        out = up + half  # elastic add
        out += residual  # residual add

        return out


class DarkNet19(nn.Module):
    """ [https://arxiv.org/pdf/1612.08242.pdf] """
    def __init__(self, num_classes=1000, init_weight=True):
        super(DarkNet19, self).__init__()

        if init_weight:
            self._initialize_weights()

        self.features = nn.Sequential(
            Conv(3, 32, 3),
            MaxPool2d(2, 2),

            Conv(32, 64, 3),
            MaxPool2d(2, 2),

            Conv(64, 128, 3),
            Conv(128, 64, 1),
            Conv(64, 128, 3),
            MaxPool2d(2, 2),

            Conv(128, 256, 3),
            Conv(256, 128, 1),
            Conv(128, 256, 3),
            MaxPool2d(2, 2),

            Conv(256, 512, 3),
            Conv(512, 256, 1),
            Conv(256, 512, 3),
            Conv(512, 256, 1),
            Conv(256, 512, 3),
            MaxPool2d(2, 2),

            Conv(512, 1024, 3),
            Conv(1024, 512, 1),
            Conv(512, 1024, 3),
            Conv(1024, 512, 1),
            Conv(512, 1024, 3),
        )

        self.classifier = nn.Sequential(
            *self.features,
            Conv(1024, num_classes, 1),
            GlobalAvgPool2d()
        )

    def forward(self, x):
        return self.classifier(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class DarkNet53(nn.Module):
    """ [https://pjreddie.com/media/files/papers/YOLOv3.pdf] """
    def __init__(self, block, num_classes=1000, init_weight=True):
        super(DarkNet53, self).__init__()
        self.num_classes = num_classes

        if init_weight:
            self._initialize_weights()

        self.block0 = nn.Sequential(
            Conv(3, 32, 3),
            Conv(32, 64, 3, 2),
        )

        self.block1 = self._make_layer(block, 64, num_blocks=1)

        self.block2 = nn.Sequential(
            Conv(64, 128, 3, 2),
            *self._make_layer(block, 128, num_blocks=2),
        )

        self.block3 = nn.Sequential(
            Conv(128, 256, 3, 2),
            *self._make_layer(block, 256, num_blocks=8),
        )

        self.block4 = nn.Sequential(
            Conv(256, 512, 3, 2),
            *self._make_layer(block, 512, num_blocks=8),
        )

        self.block5 = nn.Sequential(
            Conv(512, 1024, 3, 2),
            *self._make_layer(block, 1024, num_blocks=4),
        )

        self.classifier = nn.Sequential(
            GlobalAvgPool2d(),
            nn.Linear(1024, num_classes)
        )

    def get_bn_before_relu(self):
        bn1 = self.block1[-1]
        bn2 = self.block2[-1].layer1.bn
        bn3 = self.block4[-1].layer1.bn
        bn4 = self.block5[-1].layer1.bn
        return [bn1, bn2, bn3, bn4]

    def forward(self, x, is_feat=False):
        x = self.block0(x)
        f0 = x
        x = self.block1(x)
        f1 = x
        x = self.block2(x)
        f2 = x
        x = self.block3(x)
        f3 = x
        x = self.block4(x)
        f4 = x
        x = self.block5(x)
        f5 = x
        x = self.classifier(x)

        if is_feat:
            return [f0, f1, f2, f3, f4, f5], x
        else:
            return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    @staticmethod
    def _make_layer(block, in_channels, num_blocks):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(in_channels))
        return nn.Sequential(*layers)


class CSPDarkNet53(nn.Module):
    """ [https://pjreddie.com/media/files/papers/YOLOv3.pdf] """
    def __init__(self, block, num_classes=1000, init_weight=True):
        super(CSPDarkNet53, self).__init__()
        self.num_classes = num_classes

        if init_weight:
            self._initialize_weights()

        self.block0 = nn.Sequential(
            Conv(3, 32, 3),
            Conv(32, 64, 3, 2),
        )

        self.block1 = CSP(64, 64, num_blocks=1)

        self.block2 = nn.Sequential(
            Conv(64, 128, 3, 2),
            CSP(128, 128, num_blocks=2),
        )

        self.block3 = nn.Sequential(
            Conv(128, 256, 3, 2),
            CSP(256, 256, num_blocks=8),
        )

        self.block4 = nn.Sequential(
            Conv(256, 512, 3, 2),
            CSP(512, 512, num_blocks=8),
        )

        self.block5 = nn.Sequential(
            Conv(512, 1024, 3, 2),
            CSP(1024, 1024, num_blocks=4),
        )

        self.classifier = nn.Sequential(
            GlobalAvgPool2d(),
            nn.Linear(1024, num_classes)
        )

    def get_bn_before_relu(self):
        bn1 = self.block1.bn
        bn2 = self.block2[-1].bn
        bn3 = self.block4[-1].bn
        bn4 = self.block5[-1].bn
        return [bn1, bn2, bn3, bn4]

    def forward(self, x, is_feat=False, preact=False):
        x = self.block0(x)
        f0 = x
        x = self.block1(x)
        f1 = x
        x = self.block2(x)
        f2 = x
        x = self.block3(x)
        f3 = x
        x = self.block4(x)
        f4 = x
        x = self.block5(x)
        f5 = x
        x = self.classifier(x)

        if is_feat:
            return [f0, f1, f2, f3, f4, f5], x
        else:
            return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

def darknet19(num_classes=1000, init_weight=True):
    return DarkNet19(num_classes=num_classes, init_weight=init_weight)


def darknet53(num_classes=1000, init_weight=True):
    return DarkNet53(ResidualBlock, num_classes=num_classes, init_weight=init_weight)


def darknet53e(num_classes=1000, init_weight=True):
    """ DarkNet53 with ELASTIC block """
    return DarkNet53(Elastic, num_classes=num_classes, init_weight=init_weight)


def cspdarknet53(num_classes=1000, init_weight=True):
    """ DarkNet53 with CSP block """
    return CSPDarkNet53(CSP, num_classes=num_classes, init_weight=init_weight)

if __name__ == "__main__":
    x = torch.randn(2, 3, 32, 32)
    net = darknet53(num_classes=100)
    feats, logit = net(x, is_feat=True)

    for f in feats:
        print(f.shape, f.min().item())
    print(logit.shape)

    for m in net.get_bn_before_relu():
        if isinstance(m, nn.BatchNorm2d):
            print('pass')
        else:
            print('warning')

    from torchinfo import summary
    summary(net, input_data=x)