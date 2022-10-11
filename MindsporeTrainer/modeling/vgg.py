import math
import copy


import numpy as np
import mindspore.common.dtype as mstype
import mindspore.nn as nn
import mindspore.ops.functional as F
from mindspore.common.initializer import TruncatedNormal, initializer
from mindspore.ops import operations as P
from mindspore.ops import composite as C
import mindspore.ops as O
from mindspore.common.tensor import Tensor
from mindspore.common.parameter import Parameter
from MindsporeTrainer.utils.checkpoint import load_ckpt

class VGG16(nn.Cell):
    """
    Provide Perceptual Losse of <Perceptual Losses for Real-Time Style Transfer and Super-Resolution>.

    """

    def __init__(self, init_path=None, train=False):
        super(VGG16, self).__init__()
        self.power = P.Pow()

        self.layer_config = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
        self.layers = self._make_layer(self.layer_config)
        if init_path:
            self.layers = load_ckpt(self.layers, init_path, restore_by_prefix=False)
        if not train:
            for param in self.layers.get_parameters():
                param.requires_grad = False

    def construct(self, input):
        return self.layers(input)


    def _make_layer(self, base):
        """Make stage network of VGG."""
        pad_mode = 'same'
        padding = 0
        has_bias = False
        batch_norm = False
        initialize_mode = "XavierUniform"
        has_dropout = False

        layers = []
        in_channels = 3
        for v in base:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                weight = 'ones'
                if initialize_mode == "XavierUniform":
                    weight_shape = (v, in_channels, 3, 3)
                    weight = initializer('XavierUniform', shape=weight_shape, dtype=mstype.float32)

                conv2d = nn.Conv2d(in_channels=in_channels,
                                out_channels=v,
                                kernel_size=3,
                                padding=padding,
                                pad_mode=pad_mode,
                                has_bias=has_bias,
                                weight_init=weight)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU()]
                else:
                    layers += [conv2d, nn.ReLU()]
                in_channels = v
        return nn.SequentialCell(layers)