from .resnet import resnet8, resnet14, resnet20, resnet32, resnet44, resnet56, resnet110, resnet8x4, resnet32x4
from .resnetv2 import ResNet50
from .wrn import wrn_16_1, wrn_16_2, wrn_40_1, wrn_40_2
from .vgg import vgg19_bn, vgg16_bn, vgg13_bn, vgg11_bn, vgg8_bn
from .mobilenetv2 import mobilenetv2_6_05, mobilenetv2_6_1, mobilenetv2_6_025
from .mobilenet_v2 import mobilenet_v2_half
from .ShuffleNetv1 import ShuffleV1
from .ShuffleNetv2 import ShuffleV2
from .darknet import darknet19, darknet53, darknet53e, cspdarknet53
from .efficientnet import efficientnet_b0

model_dict = {
    'resnet8': resnet8,
    'resnet14': resnet14,
    'resnet20': resnet20,
    'resnet32': resnet32,
    'resnet44': resnet44,
    'resnet56': resnet56,
    'resnet110': resnet110,
    'resnet8x4': resnet8x4,
    'resnet32x4': resnet32x4,
    'ResNet50': ResNet50,
    'wrn_16_1': wrn_16_1,
    'wrn_16_2': wrn_16_2,
    'wrn_40_1': wrn_40_1,
    'wrn_40_2': wrn_40_2,
    'vgg8': vgg8_bn,
    'vgg11': vgg11_bn,
    'vgg13': vgg13_bn,
    'vgg16': vgg16_bn,
    'vgg19': vgg19_bn,
    'mobilenetv2_6_1': mobilenetv2_6_1,
    'mobilenetv2_6_05': mobilenetv2_6_05,
    'mobilenetv2_6_025': mobilenetv2_6_025,
    'mobilenet_v2_half': mobilenet_v2_half,
    'ShuffleV1': ShuffleV1,
    'ShuffleV2': ShuffleV2,
    'darknet19': darknet19,
    'darknet53': darknet53,
    'darknet53e': darknet53e,
    'cspdarknet53': cspdarknet53,
    'efficientnet_b0': efficientnet_b0,
}
