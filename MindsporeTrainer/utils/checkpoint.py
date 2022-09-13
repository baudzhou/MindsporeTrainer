# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# zbo@zju.edu.cn
# 2022-08-08
# ============================================================================

import os
from pyexpat import model
import shutil
from loguru import logger

import mindspore as ms
from mindspore import load_checkpoint, load_param_into_net
from mindspore.common.initializer import TruncatedNormal, initializer


def load_ckpt(model, ckpt, restore_by_prefix=True, prefix='backbone'):
    logger.info(f'Recovering last checkpoint from {ckpt}')
    params = model.get_parameters()
    # param_names = [p.name for p in params]
    param_dict = load_checkpoint(ckpt)
    # 若bert没有训练pooled out，那么backbone.bert.dense需要重新初始化
    if param_dict.get('backbone.bert.dense.weight') and param_dict['backbone.bert.dense.weight'].max() < 1e-8:
        param_dict['backbone.bert.dense.weight'].set_data(initializer(TruncatedNormal(0.02), param_dict['backbone.bert.dense.weight'].shape)) 
    if not restore_by_prefix:
        unrestored = load_param_into_net(model, param_dict)
    else:
        param_names = [p.name for p in params if prefix in p.name]
        param_dict = {f'{prefix}.{k}': v for k, v in param_dict.items()}
        param_dict = {k: v for k, v in param_dict.items() if k in param_names}
        unrestored = load_param_into_net(model, param_dict)
    # unrestored = [p for p in param_names if p not in param_dict]
    missed = [p for p in param_dict.keys() if p not in param_names]
    logger.warning(f'unrestored parameters: {unrestored}\n')
    logger.warning(f'missed parameters: {missed}\n')
    return model


def save_checkpoint(state, is_best, save_dir, model_name=""):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    filename = os.path.join(save_dir, model_name + "_ckpt.pth")
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save_dir, "best_ckpt.pth")
        shutil.copyfile(filename, best_filename)

if __name__ == '__main__':
    # from ..apps.models.resnet import ResidualBlock, ResNet

    # model = ResNet(ResidualBlock, 10)
    load_ckpt(None, 'output/resnet_cifar10/RESNETTask_3-6_781.ckpt')