# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# zbo@zju.edu.cn
# 2022-08-08
# ============================================================================

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
from mindspore.nn.transformer import CrossEntropyLoss
from MindsporeTrainer.utils.checkpoint import load_ckpt


class SoftmaxCrossEntropyExpand(nn.Cell):       # pylint: disable=missing-docstring
    def __init__(self, sparse=False):
        super(SoftmaxCrossEntropyExpand, self).__init__()
        self.exp = ops.Exp()
        self.sum = ops.ReduceSum(keep_dims=True)
        self.onehot = ops.OneHot()
        self.on_value = ms.Tensor(1.0, ms.float32)
        self.off_value = ms.Tensor(0.0, ms.float32)
        self.div = ops.RealDiv()
        self.log = ops.Log()
        self.sum_cross_entropy = ops.ReduceSum(keep_dims=False)
        self.mul = ops.Mul()
        self.mul2 = ops.Mul()
        self.mean = ops.ReduceMean(keep_dims=False)
        self.sparse = sparse
        self.max = ops.ReduceMax(keep_dims=True)
        self.sub = ops.Sub()
        self.eps = ms.Tensor(1e-24, ms.float32)

    def construct(self, logit, label):      # pylint: disable=missing-docstring
        logit_max = self.max(logit, -1)
        exp = self.exp(self.sub(logit, logit_max))
        exp_sum = self.sum(exp, -1)
        softmax_result = self.div(exp, exp_sum)
        if self.sparse:
            label = self.onehot(label, ops.shape(logit)[1], self.on_value, self.off_value)

        softmax_result_log = self.log(softmax_result + self.eps)
        loss = self.sum_cross_entropy((self.mul(softmax_result_log, label)), -1)
        loss = self.mul2(ops.scalar_to_array(-1.0), loss)
        loss = self.mean(loss, -1)

        return loss


class BertPretrainingLoss(nn.Cell):
    """
    Provide bert pre-training loss.

    Args:
        config (BertConfig): The config of BertModel.

    Returns:
        Tensor, total loss.
    """

    def __init__(self, vocab_size):
        super(BertPretrainingLoss, self).__init__()
        self.vocab_size = vocab_size
        self.onehot = P.OneHot()
        self.on_value = Tensor(1.0, mstype.float32)
        self.off_value = Tensor(0.0, mstype.float32)
        self.reduce_sum = P.ReduceSum()
        self.reduce_mean = P.ReduceMean()
        self.reshape = P.Reshape()
        self.last_idx = (-1,)
        self.neg = P.Neg()
        self.cast = P.Cast()
        self.loss_fn = CrossEntropyLoss()

    def construct(self, *sample):
        """Defines the computation performed.
            sample: ["input_ids", "input_mask", "token_type_id", "next_sentence_labels",
                     "masked_lm_positions", "masked_lm_ids", "masked_lm_weights"] + 
                    [prediction_scores, seq_relationship_score]
        """
        masked_lm_ids = sample[5]
        masked_lm_weights = sample[6]
        prediction_scores = sample[7]
        next_sentence_labels = sample[3]
        seq_relationship_score = sample[8]
        
        mask = F.cast(O.ones_like(masked_lm_ids), mstype.float32)
        masked_lm_loss = self.loss_fn(prediction_scores, masked_lm_ids.reshape((-1,)), mask.reshape((-1,)))

        # label_ids = self.reshape(masked_lm_ids, self.last_idx)
        # # label_weights = self.cast(self.reshape(masked_lm_weights, self.last_idx), mstype.float32)
        # one_hot_labels = self.onehot(label_ids, self.vocab_size, self.on_value, self.off_value)
        # self.reduce_sum(prediction_scores * one_hot_labels, self.last_idx)

        # per_example_loss = self.neg(self.reduce_sum(prediction_scores * one_hot_labels, self.last_idx))
        # numerator = self.reduce_sum(label_weights * per_example_loss, ())
        # denominator = self.reduce_sum(label_weights, ()) + self.cast(F.tuple_to_array((1e-5,)), mstype.float32)
        # masked_lm_loss = numerator / denominator

        # next_sentence_loss
        # labels = self.reshape(next_sentence_labels, self.last_idx)
        # if labels.max() >= 0:
        #     one_hot_labels = self.onehot(labels, 2, self.on_value, self.off_value)
        #     per_example_loss = self.neg(self.reduce_sum(
        #         one_hot_labels * seq_relationship_score, self.last_idx))
        #     next_sentence_loss = self.reduce_mean(per_example_loss, self.last_idx)
        # else:
        next_sentence_loss = Tensor(0.0, dtype=mstype.float32)

        # total_loss
        total_loss = masked_lm_loss + next_sentence_loss

        return total_loss


class PerceptualLoss(nn.Cell):
    """
    Provide Perceptual Losse of <Perceptual Losses for Real-Time Style Transfer and Super-Resolution>.

    """

    def __init__(self, init_path, content_loss_only=True):
        super(PerceptualLoss, self).__init__()
        self.power = P.Pow()

        self.layer_config = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
        self.content_loss_only = content_loss_only
        if content_loss_only:
            self.layer_config = [64, 64, 'M', 128, 128]
        else:
            self.layer_config = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M']
        self.layers = self._make_layer(self.layer_config)
        self.layers = load_ckpt(self.layers, init_path, restore_by_prefix=False)
        for param in self.layers.get_parameters():
            param.requires_grad = False
        self.mse = nn.MSELoss()

    def construct(self, label_img, pred_img):
        content = self.layers(label_img)
        pred_content = self.layers(pred_img)

        return self.mse(content, pred_content)


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