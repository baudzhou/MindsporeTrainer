# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# zbo@zju.edu.cn
# 2022-08-08
# ============================================================================


import argparse
from ..utils import boolean_string

__all__ = ['get_args']

def get_args():
  parser=argparse.ArgumentParser(add_help=False, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  group = parser.add_argument_group(title='Trainer', description='Parameters for the distributed trainer')
  group.add_argument('--accumulative_update',
            type=int,
            default=1,
            help="Number of updates steps to accumulate before performing a backward/update pass.")

  group.add_argument('--loss_scale_value',
            type=int,
            default=1024,
            help="initial loss scale value")

  group.add_argument('--scale_factor',
            type=int,
            default=4,
            help="loss scale factor")

  group.add_argument('--scale_window',
            type=int,
            default=1000,
            help="loss window")

  group.add_argument('--enable_global_norm',
            type=bool,
            default=False,
            help="enable global norm")
  
  group.add_argument("--report_interval",
            default=1,
            type=int,
            help="Interval steps for state report.")

  group.add_argument("--local_rank",
            type=int,
            default=0,
            help="local_rank for distributed training on gpus")

  group.add_argument('--data_workers',
            type=int,
            default=4,
            help="The workers to load data.")

  group.add_argument('--seed',
            type=int,
            default=1234,
            help="random seed for initialization")

  group.add_argument("--train_batch_size",
            default=64,
            type=int,
            help="Total batch size for training.")

  group.add_argument("--world_size",
            type=int,
            default=-1,
            help="[Internal] The world size of distributed training. Internal usage only!! To the world size of the program, you need to use environment. 'WORLD_SIZE'")

  group.add_argument("--rank",
            type=int,
            default=-1,
            help="[Internal] The rank id of current process. Internal usage only!! To the rank of the program, you need to use environment. 'RANK'")

  group.add_argument("--master_ip",
            type=str,
            default=None,
            help="[Internal] The ip address of master node. Internal usage only!! To the master IP of the program, you need to use environment. 'MASTER_ADDR'")

  group.add_argument("--master_port",
            type=str,
            default=None,
            help="[Internal] The port of master node. Internal usage only!! To the master IP of the program, you need to use environment. 'MASTER_PORT'")

  return parser
