# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# zbo@zju.edu.cn
# 2022-08-08
# ============================================================================

from collections import Sequence, Mapping
import math
from mindspore.train.callback import Callback

# def batch_apply(batch, fn):
#   if isinstance(batch, torch.Tensor):
#     return fn(batch)
#   elif isinstance(batch, Sequence):
#     return [batch_apply(x, fn) for x in batch]
#   elif isinstance(batch, Mapping):
#     return {x:batch_apply(batch[x], fn) for x in batch}
#   else:
#     raise NotImplementedError(f'Type of {type(batch)} are not supported in batch_apply')

# def batch_to(batch, device):
#   return batch_apply(batch, lambda x: x.to(device))

class LossCallBack(Callback):
    """
    Monitor the loss in training.
    If the loss in NAN or INF terminating training.
    Note:
        if per_print_times is 0 do not print loss.
    Args:
        per_print_times (int): Print loss every times. Default: 1.
    """
    def __init__(self, dataset_size=-1):
        super(LossCallBack, self).__init__()
        self._dataset_size = dataset_size
    def step_end(self, run_context):
        """
        Print loss after each step
        """
        cb_params = run_context.original_args()
        if self._dataset_size > 0:
            percent, epoch_num = math.modf(cb_params.cur_step_num / self._dataset_size)
            if percent == 0:
                percent = 1
                epoch_num -= 1
            print("epoch: {}, current epoch percent: {}, step: {}, outputs are {}"
                  .format(int(epoch_num), "%.3f" % percent, cb_params.cur_step_num, str(cb_params.net_outputs)),
                  flush=True)
        else:
            print("epoch: {}, step: {}, outputs are {}".format(cb_params.cur_epoch_num, cb_params.cur_step_num,
                                                               str(cb_params.net_outputs)), flush=True)