# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# zbo@zju.edu.cn
# 2022-08-08
# ============================================================================

import os
import psutil
import signal
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
os.environ['GLOG_v'] = '3'
import argparse
import random
import numpy as np
import subprocess
import argparse
import mindspore as ms
import mindspore.communication.management as D
from mindspore import context
from loguru import logger

from MindsporeTrainer.utils import *
from MindsporeTrainer.apps.tasks import load_tasks, get_task
from MindsporeTrainer.training import DistributedTrainer
from MindsporeTrainer.training import get_args as get_training_args
from MindsporeTrainer.optims import get_args as get_optims_args
from MindsporeTrainer.modeling.modeling_adapter import *


def kill_children(proc=None, recursive = True):
    if proc is None:
        proc = psutil.Process()
    _children = proc.children(recursive=False)
    for c in _children:
        try:
            if recursive:
                kill_children(c, recursive=recursive)
            os.kill(c.pid, signal.SIGKILL)
        except:
            pass

        for c in _children:
            try:
                c.wait(1)
            except:
                pass

class LoadTaskAction(argparse.Action):
    _registered = False
    def __call__(self, parser, args, values, option_string=None):
        setattr(args, self.dest, values)
        if not self._registered:
            load_tasks(args.task_dir, values)
            all_tasks = get_task()
            if values=="*":
                for task in all_tasks.values():
                    parser.add_argument_group(title=f'Task {task._meta["name"]}', description=task._meta["desc"])
                return

        # assert values.lower() in all_tasks, f'{values} is not registed. Valid tasks {list(all_tasks.keys())}'
        task = get_task(values)
        group = parser.add_argument_group(title=f'Task {task._meta["name"]}', description=task._meta["desc"])
        task.add_arguments(group)
        type(self)._registered = True

def build_argument_parser():
    parser = argparse.ArgumentParser(parents=[get_optims_args(), get_training_args()], formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ## Required parameters
    parser.add_argument("--task_dir",
                default=None,
                type=str,
                required=False,
                help="The directory to load customized tasks.")
    parser.add_argument("--task_name",
                default=None,
                type=str,
                action=LoadTaskAction,
                required=True,
                help="The name of the task to train. To list all registered tasks, use \"*\" as the name, e.g. \n"
                "\npython -m DeBERTa.apps.run --task_name \"*\" --help")

    parser.add_argument("--data_dir",
                default=None,
                type=str,
                required=False,
                help="The input data dir. Should contain the .tsv files (or other data files) for the task.")

    parser.add_argument("--output_dir",
                default=None,
                type=str,
                required=True,
                help="The output directory where the model checkpoints will be written.")

    parser.add_argument("--do_train",
                default=False,
                action='store_true',
                help="Whether to run training.")

    parser.add_argument("--do_eval",
                default=False,
                action='store_true',
                help="Whether to run eval on the dev set.")

    parser.add_argument("--do_predict",
                default=False,
                action='store_true',
                help="Whether to run prediction on the test set.")

    parser.add_argument("--eval_batch_size",
                default=32,
                type=int,
                help="Total batch size for eval.")

    parser.add_argument("--predict_batch_size",
                default=32,
                type=int,
                help="Total batch size for prediction.")

    parser.add_argument('--tag',
                type=str,
                default='final',
                help="The tag name of current prediction/runs.")

    parser.add_argument('--debug',
                default=False,
                action='store_true',
                help="Whether to cache cooked binary features")

    parser.add_argument("--num_train_epochs",
                default=1,
                type=int,
                help="Total number of training epochs to perform.")

    parser.add_argument('--device_target', 
                        type=str, 
                        default='Ascend', 
                        choices=['Ascend', 'GPU'],
                        help='device where the code will be implemented. (Default: Ascend)')
    parser.add_argument('--run_mode', 
                        type=str, 
                        default='GRAPH', 
                        choices=['GRAPH', 'PY'],
                        help='0: GRAPH_MODE, 1: PY_NATIVE_MODE')
    parser.add_argument("--distribute", 
                        default=False, 
                        action='store_true',
                        help="Run distribute, default is false.")

    parser.add_argument("--device_id", type=str, default="0", help="Device id, default is 0.")
    parser.add_argument("--device_num", type=int, default=1, help="Use device nums, default is 1.")
    parser.add_argument("--enable_save_ckpt", type=str, default="true", choices=["true", "false"],
                        help="Enable save checkpoint, default is true.")
    parser.add_argument("--enable_lossscale", type=str, default="true", choices=["true", "false"],
                        help="Use lossscale or not, default is not.")
    parser.add_argument("--do_shuffle", type=str, default="true", choices=["true", "false"],
                        help="Enable shuffle for dataset, default is true.")
    parser.add_argument("--enable_data_sink",
                        default=False,
                        action='store_true',
                        help="Enable data sink, default is false.")
    parser.add_argument("--data_sink_steps", type=int, default="1", help="Sink steps for each epoch, default is 1.")
    parser.add_argument("--accumulation_steps", type=int, default="1",
                        help="Accumulating gradients N times before weight update, default is 1.")
    parser.add_argument("--allreduce_post_accumulation", type=str, default="true", choices=["true", "false"],
                        help="Whether to allreduce after accumulation of N steps or after each step, default is true.")
    # parser.add_argument("--save_checkpoint_path", type=str, default="", help="Save checkpoint path")
    parser.add_argument("--load_checkpoint_path", type=str, default="", help="Load checkpoint file path")
    parser.add_argument("--save_eval_steps", type=int, default=1000, help="Save checkpoint and evaluate steps, "
                                                                                "default is 1000.")
    parser.add_argument("--train_steps", type=int, default=-1, help="Training Steps, default is -1, "
                                                                    "meaning run all steps according to epoch number.")
    parser.add_argument("--save_checkpoint_num", type=int, default=1, help="Save checkpoint numbers, default is 1.")

    # parser.add_argument("--schema_dir", type=str, default="", help="Schema path, it is better to use absolute path")
    parser.add_argument("--enable_graph_kernel", type=str, default="auto", choices=["auto", "true", "false"],
                        help="Accelerate by graph kernel, default is auto.")
    parser.add_argument("--save_graphs", 
                        default=False,
                        action='store_true',
                        help="Whether to save graphs")
    parser.add_argument("--thor", 
                        default=False,
                        action='store_true',
                        help="Whether to convert model to thor optimizer")
    return parser


def main(args):
    if not args.do_train and not args.do_eval and not args.do_predict:
        raise ValueError("At least one of `do_train` or `do_eval` or `do_predict` must be True.")
    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.accumulation_steps > 1:
        logger.info("accumulation steps: {}".format(args.accumulation_steps))
        logger.info("global batch size: {}".format(args.train_batch_size))
        if args.enable_data_sink == "true":
            args.data_sink_steps *= args.accumulation_steps
            logger.info("data sink steps: {}".format(args.data_sink_steps))
        if args.enable_save_ckpt == "true":
            args.save_eval_steps *= args.accumulation_steps
            logger.info("save checkpoint steps: {}".format(args.save_eval_steps))
        args.train_batch_size = args.train_batch_size // args.accumulation_steps
        if args.train_steps > 0:
            args.train_steps = args.train_steps * args.accumulation_steps

    task = get_task(args.task_name)(args)
    trainer = DistributedTrainer(args, task)
    trainer.run()


def set_context(args):
    """set_parameter"""
    target = args.device_target
    if target == "CPU":
        args.distribute = False
    ms.context.set_context(reserve_class_name_in_scope=False)
    # init context
    if args.run_mode == 'GRAPH':
        if target == "Ascend":
            rank_save_graphs_path = os.path.join(args.output_dir, "soma", str(os.getenv('DEVICE_ID')))
            ms.context.set_context(mode=ms.context.GRAPH_MODE, 
                                   device_target=target, 
                                   save_graphs=args.save_graphs,
                                   save_graphs_path=args.output_dir, 
                                   device_id=int(args.device_id))
        else:
            ms.context.set_context(mode=ms.context.GRAPH_MODE, 
                                   device_target=target, 
                                   save_graphs=args.save_graphs, 
                                   save_graphs_path=args.output_dir, 
                                   device_id=int(args.device_id))

    else:
        ms.context.set_context(mode=ms.context.PYNATIVE_MODE, 
                               device_target=target, 
                               save_graphs=args.save_graphs, 
                               save_graphs_path=args.output_dir, 
                               device_id=int(args.device_id))
    if args.enable_graph_kernel == "true":
        if args.device_target == 'GPU':
            context.set_context(enable_graph_kernel=True)
        else:
            logger.warning('Graph kernel only supports GPU back-end now, run with graph kernel off.')
    if args.distribute:
        if target == "Ascend":
            context.reset_auto_parallel_context()
            ms.context.set_auto_parallel_context(device_num=args.device_num, 
                                                 parallel_mode=ms.context.ParallelMode.DATA_PARALLEL,
                                                 gradients_mean=False)

            D.init()
        # GPU target
        else:
            D.init()
            context.reset_auto_parallel_context()
            ms.context.set_auto_parallel_context(device_num=D.get_device_num(),
                                                parallel_mode=ms.context.ParallelMode.DATA_PARALLEL,
                                                gradients_mean=False)


def launch():
    argv = sys.argv

    parser = build_argument_parser()
    parser.parse_known_args()

    args_opt = parser.parse_args()
    os.makedirs(args_opt.output_dir, exist_ok=True)
    device_num = int(os.getenv('RANK_SIZE', '1'))
    args_opt.device_num = device_num
    logger.info(args_opt)
    if len(args_opt.device_id) > 1:
        device_id = args_opt.device_id.split(',')
        device_id = [int(id) for id in device_id]
        if len(device_id) != device_num:
            if device_num < 8:
                raise f'device id: {device_id} should equal to rank size{device_num}'
        args_opt.device_id = device_id
    else:
        # 若device id 只有一个，rank size > 1，那么device id 作为起始设备编号
        device_id = int(args_opt.device_id)
        device_id = [device_id + i for i in range(device_num)]

    if args_opt.distribute:
        if args_opt.device_target == 'Ascend':
            child_process = []
            if args_opt.rank == -1:
                try:
                    for i, id in enumerate(device_id):
                        rank_id = id + args_opt.local_rank
                        os.putenv('RANK_ID', str(rank_id))
                        os.putenv('DEVICE_ID', str(id))
                        child = subprocess.Popen(['python'] + argv + [f'--rank={rank_id}'] + [f'--device_id={id}'])
                        child_process.append(child)
                    for c in child_process:
                        c.wait()
                except Exception as e:
                    try:
                        for c in child_process:
                            c.kill()
                        logger.exception(f'Uncatched exception happened during execution.')
                        import atexit
                        atexit._run_exitfuncs()
                    except:
                        pass
            else:
                set_context(args_opt)
                try:
                    main(args_opt)
                except Exception as ex:
                    try:
                        logger.exception(f'Uncatched exception happened during execution.')
                        import atexit
                        atexit._run_exitfuncs()
                    except:
                        pass
                    kill_children()
                    os._exit(-1)

        else:
            os.environ["MS_WORKER_NUM"] == device_num
            child_process = []
            if args_opt.rank == -1:
                os.putenv('MS_ROLE', "MS_WORKER")
                for id in device_id:
                    rank_id = id + args_opt.local_rank
                    os.putenv('RANK_ID', str(rank_id))
                    os.putenv('DEVICE_ID', str(id))
                    child = subprocess.Popen(['python'] + argv + [f'--rank={rank_id}'] + [f'--device_id={id}'])
                    child_process.append(child)
            else:
                id = 0
        if args_opt.rank == -1:
            os.environ["MS_ROLE"] = "MS_SCHED"
        set_context(args_opt)
        try:
            main(args_opt)
        except Exception as ex:
            try:
                logger.exception(f'Uncatched exception happened during execution.')
                import atexit
                atexit._run_exitfuncs()
            except:
                pass
            kill_children()
            os._exit(-1)

    else:
        args_opt.rank = 0
        args_opt.local_rank = 0
        args_opt.device_id = device_id[0]
        set_context(args_opt)
        main(args_opt)

