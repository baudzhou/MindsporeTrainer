# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# zbo@zju.edu.cn
# 2022-08-08
# ============================================================================

import os
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.dataset as ds
import mindspore.dataset.vision as vision
import mindspore.dataset.transforms as transforms
from mindspore.communication import init, get_rank, get_group_size
from mindspore.nn import Momentum
from mindspore.nn.metrics import Metric
from mindspore.ops.composite import GradOperation
import mindspore as ms
from MindsporeTrainer.apps.models.resnet import ResidualBlock, ResNet, resnet50
from mindspore.parallel._ps_context import _is_role_pserver, _is_role_sched

ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target='Ascend', device_id=0)
# init("nccl")

def create_dataset(data_path, repeat_num=1, batch_size=32, rank_id=0, rank_size=1):     # pylint: disable=missing-docstring
    resize_height = 224
    resize_width = 224
    rescale = 1.0 / 255.0
    shift = 0.0

    # get rank_id and rank_size
    # rank_id = get_rank()
    # rank_size = get_group_size()
    indices = list(range(128))
    sampler = ds.SubsetRandomSampler(indices=indices)
    data_set = ds.Cifar10Dataset(data_path)

    # define map operations
    random_crop_op = vision.c_transforms.RandomCrop((32, 32), (4, 4, 4, 4))
    random_horizontal_op = vision.c_transforms.RandomHorizontalFlip()
    resize_op = vision.c_transforms.Resize((resize_height, resize_width))
    rescale_op = vision.c_transforms.Rescale(rescale, shift)
    normalize_op = vision.c_transforms.Normalize((0.4465, 0.4822, 0.4914), (0.2010, 0.1994, 0.2023))
    changeswap_op = vision.c_transforms.HWC2CHW()
    type_cast_op = transforms.c_transforms.TypeCast(ms.int32)

    c_trans = [random_crop_op, random_horizontal_op]
    c_trans += [resize_op, rescale_op, normalize_op, changeswap_op]

    # apply map operations on images
    data_set = data_set.map(operations=type_cast_op, input_columns="label")
    data_set = data_set.map(operations=c_trans, input_columns="image")

    # apply shuffle operations
    data_set = data_set.shuffle(buffer_size=10)

    # apply batch operations
    data_set = data_set.batch(batch_size=batch_size, drop_remainder=True)

    # apply repeat operations
    data_set = data_set.repeat(repeat_num)

    return data_set


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

class BertMetric(Metric):
    """
    The metric of bert network.
    Args:
        batch_size (int): The batchsize of each device.
    """
    def __init__(self):
        super(BertMetric, self).__init__()
        self.clear()

    def clear(self):
        self.logits = None
        self.labels = None

    def update(self, *inputs):
        logits, labels = inputs
        if self.logits is None:
            self.logits = logits.argmax(-1)
            self.labels = labels
        else:
            self.logits = ms.ops.Concat()([self.logits, logits.argmax(-1)])
            self.labels = ms.ops.Concat()([self.labels, labels])

    def eval(self):
        acc = self.logits == self.labels
        acc = acc.sum()
        total = self.labels.size
        acc = acc.astype('float') / total
        return acc

import time
class StepTimeMonitor(ms.train.callback.Callback):
    def __init__(self, data_size=None):
        super(StepTimeMonitor, self).__init__()
        self.data_size = data_size
        self.step_time = 0
        self.steps = 0
        self.results = []
    
    def step_begin(self, run_context):
        """
        Called before each step beginning.

        Args:
            run_context (RunContext): Include some information of the model.
        """
        self.step_time = time.time()

    def step_end(self, run_context):
        """
        Called after each step finished.

        Args:
            run_context (RunContext): Include some information of the model.
        """
        self.steps += 1
        self.results.append(run_context.original_args().net_outputs)
        print(f'step time: {time.time() - self.step_time}')


def test_train_cifar(epoch_size=10):        # pylint: disable=missing-docstring
    # ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.AUTO_PARALLEL, gradients_mean=True)
    loss_cb = ms.train.callback.LossMonitor()
    time_cb = ms.train.callback.TimeMonitor(data_size=1)
    batch_size = 256
    num_classes = 10
    data_path = 'data/tasks/cifar-10-batches-bin'
    dataset = create_dataset(data_path, batch_size=batch_size)

    net = resnet50(32,10).set_train()# ResNet(ResidualBlock, num_classes, batch_size)
    net = ResNet(ResidualBlock, num_classes, batch_size)
    scale_factor = 4
    scale_window = 3000
    # loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    loss = SoftmaxCrossEntropyExpand(sparse=True)
    opt = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), 0.01, 0.9)
    loss_scale_manager = ms.DynamicLossScaleManager(scale_factor, scale_window)
    net = ms.amp.build_train_network(net, opt, level='O3', loss_scale_manager=loss_scale_manager, loss_fn=loss)

    
    grad_op = GradOperation(get_by_list=True, sens_param=True)
    batch = next(dataset.create_tuple_iterator())
    weigths = [w for w in net.get_parameters()]
    logits = net(*batch)
    grads = grad_op(net, weigths)(*batch, ms.Tensor(10, dtype=ms.dtype.float32))
    model = ms.Model(net, loss_fn=loss, optimizer=opt, amp_level='O0', metrics={'acc'})
    model.train(1, dataset, callbacks=[loss_cb, StepTimeMonitor()], dataset_sink_mode=False)
    # m = StepTimeMonitor()
    # metrics = model.eval(dataset, callbacks=[m])
    # print(metrics)

if __name__ == '__main__':
    # _is_role_sched()
    test_train_cifar()