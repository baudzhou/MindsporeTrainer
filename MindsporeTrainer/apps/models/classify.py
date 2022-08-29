# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# zbo@zju.edu.cn
# 2022-08-08
# ============================================================================

import math
import copy
from collections import Sequence
import mindspore

import numpy
import mindspore.numpy as np
import mindspore.common.dtype as mstype
import mindspore.nn as nn
import mindspore.ops.functional as F
from mindspore.common.initializer import TruncatedNormal, initializer
from mindspore.ops import operations as P
from mindspore.ops import composite as C
import mindspore.ops as O
from mindspore.common.tensor import Tensor
from mindspore.common.parameter import Parameter

class BertForClassify(nn.Cell):
    """
    Provide loss through network.

    Args:
        is_training (bool): Specifies whether to use the training mode.
        use_one_hot_embeddings (bool): Specifies whether to use one-hot for embeddings. Default: False.

    Returns:
        Tensor, the loss of the network.
    """

    def __init__(self, backbone, num_labels, return_all=False, fp16=False, dropout=0.1):
        super(BertForClassify, self).__init__()
        self.bert = backbone
        self.classifier = nn.Dense(self.bert.hidden_size, num_labels, has_bias=False)
        # self.loss_fn = nn.SoftmaxCrossEntropyWithLogits()
        self.dropout = nn.Dropout(1 - dropout)
        self.fp16 = fp16
        self.return_all = return_all

    def construct(self, *sample):
        input_ids, input_mask, token_type_id = sample[: 3]
        
        prediction_scores = self.bert(input_ids, token_type_id, input_mask)
        pooled_out = F.cast(prediction_scores[1], mstype.float32)
        logits = self.classifier(self.dropout(pooled_out))
        # total_loss = self.loss_fn(logits, label)
        # total_loss = F.cast(total_loss, mstype.float32)
        return (logits,)


class ClsLoss(nn.SoftmaxCrossEntropyWithLogits):
    def __init__(self, sparse=True, reduction='none'):
        super().__init__(sparse=sparse, reduction=reduction)
    
    def construct(self, *inputs):
        logits = inputs[-1]
        labels = inputs[-2]
        return super().construct(logits, labels)