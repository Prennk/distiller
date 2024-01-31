import torch
from torch import nn
from torch.hub import load_state_dict_from_url

model_urls = {
    'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
}


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
            
        layers.extend([
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),

            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup), 
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1.0, inverted_residual_setting=None, round_nearest=8):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                # 208,208,32 -> 208,208,16
                [1, 16, 1, 1],
                # 208,208,16 -> 104,104,24
                [6, 24, 2, 2],
                # 104,104,24 -> 52,52,32
                [6, 32, 3, 2],

                # 52,52,32 -> 26,26,64
                [6, 64, 4, 2],
                # 26,26,64 -> 26,26,96
                [6, 96, 3, 1],
                
                # 26,26,96 -> 13,13,160
                [6, 160, 3, 2],
                # 13,13,160 -> 13,13,320
                [6, 320, 1, 1],
            ]

        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)

        # 416,416,3 -> 208,208,32
        features = [ConvBNReLU(3, input_channel, stride=2)]

        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel

        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
        self.features = nn.Sequential(*features)

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    # def get_bn_before_relu(self):
    #     bn1 = self.features[3].conv[-1]
    #     bn2 = self.features[6].conv[-1]
    #     bn3 = self.features[13].conv[-1]
    #     bn4 = self.features[17].conv[-1]
    #     return [bn1, bn2, bn3, bn4]

    def forward(self, x):
        x = self.features(x)
        x = x.mean([2, 3])
        x = self.classifier(x)
        return x
    
        # x = self.features[0](x)
        # f0 = x

        # x = self.features[1:4](x)
        # f1 = x

        # x = self.features[4:7](x)
        # f2 = x

        # x = self.features[7:14](x)
        # f3 = x
        
        # x = self.features[14:18](x)
        # f4 = x

        # x = self.features[18](x)
        # x = x.mean([2, 3])
        # f5 = x

        # x = self.classifier(x)

        # if is_feat:
        #     return [f0, f1, f2, f3, f4, f5], x
        # else:
        #     return x

def mobilenet_v2(pretrained=False, progress=True):
    model = MobileNetV2(width_mult=1.0)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['mobilenet_v2'], model_dir="model_data",
                                              progress=progress)
        model.load_state_dict(state_dict)

    return model

def mobilenet_v2_half(num_classes=1000, pretrained=False, progress=True):
    model = MobileNetV2(width_mult=0.5, num_classes=num_classes)
    if pretrained:
        raise ValueError('No Pretrained!!!!')

    return model

class MobileNetV2_half(nn.Module):
    def __init__(self, num_classes=1000, pretrained = False):
        super(MobileNetV2_half, self).__init__()
        self.model = mobilenet_v2_half(num_classes=num_classes, pretrained=pretrained)

    def forward(self, x, is_feat=False, preact=False):
        return self.model(x)

        # out3 = self.model.features[:7](x)
        # out4 = self.model.features[7:14](out3)
        # out5 = self.model.features[14:18](out4)
        # return out3, out4, out5

class MobileNetV2_half_Backbone(nn.Module):
    def __init__(self, num_classes=1000):
        super(MobileNetV2_half_Backbone, self).__init__()
        self.backbone = MobileNetV2_half(num_classes=num_classes, pretrained=False)

    def forward(self, x, is_feat=False, preact=False):
        x = self.backbone.model.features[0](x)
        f0 = x

        x = self.backbone.model.features[1:4](x)
        f1 = x

        x = self.backbone.model.features[4:7](x)
        f2 = x

        x = self.backbone.model.features[7:14](x)
        f3 = x
        
        x = self.backbone.model.features[14:18](x)
        f4 = x

        x = self.backbone.model.features[18](x)
        x = x.mean([2, 3])
        f5 = x

        x = self.backbone.model.classifier(x)

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
    net = mobilenet_v2_half()

    feats, logit = net(x, is_feat=True)
    for f in feats:
        print(f.shape, f.min().item())
    print(logit.shape)

    for m in net.get_bn_before_relu():
        if isinstance(m, nn.BatchNorm2d):
            print('pass')
        else:
            print('warning')