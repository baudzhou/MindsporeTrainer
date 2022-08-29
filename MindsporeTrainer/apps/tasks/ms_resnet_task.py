# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# zbo@zju.edu.cn
# 2022-08-08
# ============================================================================

from collections import OrderedDict

import mindspore.dataset as ds
import mindspore.dataset.vision as vision
import mindspore.dataset.transforms as transforms
from mindspore.communication import get_rank, get_group_size
from mindspore.nn import SoftmaxCrossEntropyWithLogits
import mindspore as ms

from MindsporeTrainer.task import Task
from MindsporeTrainer.task  import register_task
from MindsporeTrainer.utils.metrics import *

@register_task(name="RESNET", desc="a task demo for resnet on cifar10")
class RESNETTask(Task):
    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)
        self.args = args
        self.data_dir = args.data_dir
        self.metric = {'acc': MSAucuracy()}
        self.main_metric = 'acc'
        self.optimizer_name = 'Momentum'

    def train_data(self, **kwargs):
        return self.create_dataset(usage='train', batch_size=self.train_batch_size, rank_id=self.rank_id, rank_size=self.rank_size)

    def eval_data(self, **kwargs):
        return self.create_dataset(usage='test', batch_size=self.predict_batch_size, rank_id=self.rank_id, rank_size=self.rank_size)

    def test_data(self, **kwargs):
        return self.create_dataset(usage='test', batch_size=self.eval_batch_size, rank_id=self.rank_id, rank_size=self.rank_size)

    def get_metrics(self, **kwargs):
        """Calcuate metrics based on prediction results"""
        return {'acc': MSAucuracy()}

    def get_eval_fn(self, **kwargs):
        '''
        we use a default eval function
        '''
        return None

    def get_feature_fn(self, **kwargs):
        '''
        we do not need a feature function here
        '''
        def _example_to_feature(**kwargs):
            return None
        return _example_to_feature

    def get_labels(self):
        """labels of this task."""
        return [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    def create_dataset(self, usage='train', repeat_num=1, batch_size=32, rank_id=0, rank_size=1):
        '''
        build the dataset
        '''
        resize_height = 224
        resize_width = 224
        rescale = 1.0 / 255.0
        shift = 0.0

        data_set = ds.Cifar10Dataset(self.data_dir, usage=usage, num_shards=rank_size, shard_id=rank_id, num_parallel_workers=4)

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

    def get_model(self):
        '''
        build the model
        '''
        from MindsporeTrainer.apps.models.resnet import ResidualBlock, ResNet

        return ResNet(ResidualBlock, len(self.get_labels()))

    def get_loss(self, *args, **kwargs):
        '''
        apply cross entropy loss
        '''
        return SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

    def get_eval_head(self, *args, **kwargs):
        from MindsporeTrainer.modeling.layers import ClsEvalHead
        return ClsEvalHead(len(self.get_labels()))

    def get_opt_fn(self, *args, **kwargs):
        '''
        we use a default optimizer
        '''
        return None