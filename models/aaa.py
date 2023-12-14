import torch
import torch.nn as nn
from torch.nn import MaxPool2d, functional as F

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

class DarkNet53(nn.Module):
    """ [https://pjreddie.com/media/files/papers/YOLOv3.pdf] """
    def __init__(self, block, num_classes=1000, init_weight=True):
        super(DarkNet53, self).__init__()
        self.num_classes = num_classes

        if init_weight:
            self._initialize_weights()

        self.block1 = nn.Sequential(
            Conv(3, 32, 3),
            Conv(32, 64, 3, 2),
        )

        self.block2 = self._make_layer(block, 64, num_blocks=1)

        self.block3 = nn.Sequential(
            Conv(64, 128, 3, 2),
            *self._make_layer(block, 128, num_blocks=2),
        )

        self.block4 = nn.Sequential(
            Conv(128, 256, 3, 2),
            *self._make_layer(block, 256, num_blocks=8),
        )

        self.block5 = nn.Sequential(
            Conv(256, 512, 3, 2),
            *self._make_layer(block, 512, num_blocks=8),
        )

        self.block6 = nn.Sequential(
            Conv(512, 1024, 3, 2),
            *self._make_layer(block, 1024, num_blocks=4),
        )

        self.classifier = nn.Sequential(
            GlobalAvgPool2d(),
            nn.Linear(1024, num_classes)
        )

    def get_bn_before_relu(self):
        bn1 = self.block1[1].bn
        bn2 = self.block2[-1].layer1.bn
        bn3 = self.block4[-1].layer1.bn
        bn4 = self.block6[-1].layer1.bn
        return [bn1, bn2, bn3, bn4]

    def forward(self, x, is_feat=False):
        x = self.block1(x)
        f0 = x
        x = self.block2(x)
        f1 = x
        x = self.block3(x)
        f2 = x
        x = self.block4(x)
        f3 = x
        x = self.block5(x)
        f4 = x
        x = self.block6(x)
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
    
def darknet53(num_classes=1000, init_weight=True):
    return DarkNet53(ResidualBlock, num_classes=num_classes, init_weight=init_weight)

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
