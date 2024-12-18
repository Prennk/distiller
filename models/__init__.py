from .resnet import resnet8, resnet14, resnet20, resnet32, resnet44, resnet56, resnet110, resnet8x4, resnet32x4
from .resnetv2 import ResNet50
from .wrn import wrn_16_1, wrn_16_2, wrn_40_1, wrn_40_2
from .vgg import vgg19_bn, vgg16_bn, vgg13_bn, vgg11_bn, vgg8_bn
from .mobilenetv2 import mobilenetv2_6_05, mobilenetv2_6_1, mobilenetv2_6_025
# from .mobilenetv2 import MobileNetV2_half_Backbone
from .ShuffleNetv1 import ShuffleV1
from .ShuffleNetv2 import ShuffleV2
from .darknet import darknet19, darknet53, darknet53e, cspdarknet53
from .CSPdarknet import CSPDarkNet53Backbone
from .efficientnet import efficientnet_b0
from .repvit import repvit_m0_6, repvit_m0_9, repvit_m1_0, repvit_m1_1, repvit_m1_5, repvit_m2_3

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
    # 'mobilenetv2_half_backbone': MobileNetV2_half_Backbone,
    'ShuffleV1': ShuffleV1,
    'ShuffleV2': ShuffleV2,
    'darknet19': darknet19,
    'darknet53': darknet53,
    'darknet53e': darknet53e,
    'cspdarknet53': cspdarknet53,
    'cspdarknet53_backbone': CSPDarkNet53Backbone,
    'efficientnet_b0': efficientnet_b0,
    'repvit_m0_6': repvit_m0_6,
    'repvit_m0_9': repvit_m0_9,
    'repvit_m1_0': repvit_m1_0,
    'repvit_m1_1': repvit_m1_1,
    'repvit_m1_5': repvit_m1_5,
    'repvit_m2_3': repvit_m2_3,
}
