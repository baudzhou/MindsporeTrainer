# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# zbo@zju.edu.cn
# 2022-08-08
# ============================================================================

from mindspore import context
from mindspore.nn.optim import Lamb, Momentum, AdamWeightDecay
from .utils import LearningRate

from .args import get_args
from .adam import *


def get_optimizer(args_opt, network, opt_name='AdamWeightDecay'):
    """get ernie optimizer, support Lamb, Momentum, AdamWeightDecay."""
    params = network.trainable_params()
    decay_filter = lambda x: 'layernorm' not in x.name.lower() and 'bias' not in x.name.lower()
    decay_params = list(filter(decay_filter, params))
    other_params = list(filter(lambda x: not decay_filter(x), params))
    group_params = [{'params': decay_params, 'weight_decay': 1e-2},
                    {'params': other_params, 'weight_decay': 0.0},
                    {'order_params': params}]
    opt_overflow = False
    if opt_name == 'Lamb':
        lr_schedule = LearningRate(learning_rate=args_opt.learning_rate,
                                        end_learning_rate=0.0,
                                        warmup_steps=int(args_opt.warmup * args_opt.train_steps),
                                        decay_steps=args_opt.train_steps,
                                        power=2.0)

        optimizer = Lamb(group_params, learning_rate=lr_schedule, eps=1e-8)
    elif opt_name == 'Momentum':
        optimizer = Momentum(filter(lambda x: x.requires_grad, network.get_parameters()), learning_rate=args_opt.learning_rate,  #network.trainable_params()
                             momentum=0.9)
    elif opt_name == 'AdamWeightDecay':
        lr_schedule = LearningRate(learning_rate=args_opt.learning_rate,
                                        end_learning_rate=0.0,
                                        warmup_steps=int(args_opt.warmup * args_opt.train_steps),
                                        decay_steps=args_opt.train_steps,
                                        power=5.0)

        if args_opt.enable_lossscale == "true" and args_opt.device_target == 'GPU':
            optimizer = AdamWeightDecayX(group_params, learning_rate=lr_schedule, eps=1e-6)
            opt_overflow = True
        elif context.get_context("mode") == context.PYNATIVE_MODE and args_opt.device_target == 'GPU':
            optimizer = AdamWeightDecayOp(group_params, learning_rate=lr_schedule, eps=1e-6)
        else:
            optimizer = AdamWeightDecay(group_params, learning_rate=lr_schedule, eps=1e-6)
            opt_overflow = True
    else:
        raise ValueError("Don't support optimizer {}, only support [Lamb, Momentum, AdamWeightDecay, Thor]".
                         format(opt_name))
    return optimizer, opt_overflow
