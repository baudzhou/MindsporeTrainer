# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# zbo@zju.edu.cn
# 2022-08-08
# ============================================================================

from ast import Not
import time, os
import mindspore as ms
import numpy as np
from loguru import logger
from mindspore import Tensor
from mindspore.train.callback import Callback, ModelCheckpoint, _set_cur_net
import mindspore.context as context
from mindspore.train.serialization import save_checkpoint, _save_graph
from mindspore.context import ParallelMode
from mindspore.ops import operations as P
from mindspore import load_checkpoint, load_param_into_net

from MindsporeTrainer.modeling.modeling_adapter import ModelForEval, AllGather

class StepTimeMonitor(Callback):
    def __init__(self, data_size=None):
        super(StepTimeMonitor, self).__init__()
        self.data_size = data_size
        self.step_time = 0
    
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
        print(f'step time: {time.time() - self.step_time}')

class LossMoniter(ms.train.callback.LossMonitor):
    def step_end(self, run_context):
        """
        Print training loss at the end of step.

        Args:
            run_context (RunContext): Include some information of the model.
        """
        cb_params = run_context.original_args()
        if not 'cur_epoch_num' in cb_params:
            cb_params.cur_epoch_num = 0

        loss = cb_params.net_outputs

        if isinstance(loss, (tuple, list)):
            if isinstance(loss[0], Tensor) and isinstance(loss[0].asnumpy(), np.ndarray):
                loss = loss[0]

        if isinstance(loss, Tensor) and isinstance(loss.asnumpy(), np.ndarray):
            loss = float(np.mean(loss.asnumpy()))

        cur_step_in_epoch = (cb_params.cur_step_num - 1) % cb_params.batch_num + 1

        if isinstance(loss, float) and (np.isnan(loss) or np.isinf(loss)):
            raise ValueError("epoch: {} step: {}. Invalid loss, terminating training.".format(
                cb_params.cur_epoch_num, cur_step_in_epoch))

        # #In disaster recovery scenario, the cb_params.cur_step_num may be rollback to previous step
        # # and be less than self._last_print_time, so self._last_print_time need to be updated.
        # if self._per_print_times != 0 and (cb_params.cur_step_num <= self._last_print_time):
        #     while cb_params.cur_step_num <= self._last_print_time:
        #         self._last_print_time -=\
        #             max(self._per_print_times, cb_params.batch_num if cb_params.dataset_sink_mode else 1)

        if cb_params.cur_step_num % self._per_print_times ==0 :
            # self._last_print_time = cb_params.cur_step_num
            logger.info("Evaluating: steps: %s, loss is %s" % (cur_step_in_epoch, loss))

class EvalCallBack(Callback):
    """
    Evaluate after a certain amount of training samples.
    Args:
        model (Model): The network model.
        eval_ds (Dataset): The eval dataset.
        global_batch (int): The batchsize of the sum of all devices.
        eval_samples (int): The number of eval interval samples.
    """
    def __init__(self, model, trainer_state, eval_ds, global_batch, eval_fn, train_steps,
                    eval_steps=0, eval_samples=None, rank=0):
        super(EvalCallBack, self).__init__()
        self.model = model
        self.eval_ds = eval_ds
        self.global_batch = global_batch
        self.eval_samples = eval_samples
        self.last_eval_step = 0
        self.eval_steps = eval_steps
        self.trainer_state = trainer_state
        self.eval_fn = eval_fn
        self.rank = rank
        self._last_triggered_step = 0
        self.train_steps = train_steps


    def step_end(self, run_context):
        if self.eval_steps == 0:
            return
        else:
            if (run_context.original_args().cur_step_num  >= self._last_triggered_step + self.eval_steps):
                self.evaluate(run_context)

    def epoch_end(self, run_context):
        """
        Evaluate after training a certain number of samples.
        """
        if self.eval_steps > 0:
            return
        else:
            self.evaluate(run_context)
    
    def evaluate(self, run_context):
        cb_params = run_context.original_args()
        self._last_triggered_step = cb_params.cur_step_num 

        metric = self.eval_fn(self.model, 'eval', cb_params.cur_step_num)
        if self.trainer_state.neg_metric:
            metric = -metric
        if metric > self.trainer_state.best_metric:
            self.trainer_state.best_metric = metric
            self.trainer_state.best_steps = cb_params.cur_step_num

    def end(self, run_context):
        """
        Called once after network training.

        Args:
            run_context (RunContext): Include some information of the model.
        """
        self.evaluate(run_context)


class SummaryCallback(Callback):
    def __init__(self, summary_dir):
        self._summary_dir = summary_dir

    def __enter__(self):
        # init you summary record in here, when the train script run, it will be inited before training
        self.summary_record =  ms.train.summary.SummaryRecord(self._summary_dir)
        return self

    def __exit__(self, *exc_args):
        # Note: you must close the summary record, it will release the process pool resource
        # else your training script will not exit from training.
        self.summary_record.close()

    def step_end(self, run_context):
        cb_params = run_context.original_args()

        # create a confusion matric image, and record it to summary file
        # self.summary_record.add_value('image', 'image0', cb_params.train_dataset_element[0])
        self.summary_record.add_value('scalar', 'loss', cb_params.net_outputs)
        self.summary_record.record(cb_params.cur_step_num)


class StateCallback(Callback):
    def __init__(self, trainer_state, optimizer, train_steps, report_steps=1, summary_dir=None):
        self.trainer_state = trainer_state
        self.optimizer = optimizer
        self.report_steps = report_steps
        self.summary_dir = summary_dir
        self.summary_record = None
        self.train_steps = train_steps
        self.cur_step_num = 0

    def step_end(self, run_context):
        cb_params = run_context.original_args()
        if not hasattr(cb_params, 'cur_epoch_num'):
            cb_params.cur_epoch_num = 0
        self.cur_step_num = cb_params.cur_epoch_num
        loss = cb_params.net_outputs

        if isinstance(loss, (tuple, list)):
            if isinstance(loss[0], Tensor) and isinstance(loss[0].asnumpy(), np.ndarray):
                loss = loss[0]

        if isinstance(loss, Tensor) and isinstance(loss.asnumpy(), np.ndarray):
            loss = float(np.mean(loss.asnumpy()))

        cur_step_in_epoch = (cb_params.cur_step_num - 1) % cb_params.batch_num + 1

        if isinstance(loss, float) and (np.isnan(loss) or np.isinf(loss)):
            raise ValueError("epoch: {} step: {}. Invalid loss, terminating training.".format(
                cb_params.cur_epoch_num, cur_step_in_epoch))

        #In disaster recovery scenario, the cb_params.cur_step_num may be rollback to previous step
        # and be less than self._last_print_time, so self._last_print_time need to be updated.
        # if self._per_print_times != 0 and (cb_params.cur_step_num <= self._last_print_time):
        #     while cb_params.cur_step_num <= self._last_print_time:
        #         self._last_print_time -=\
        #             max(self._per_print_times, cb_params.batch_num if cb_params.dataset_sink_mode else 1)

        # if self._per_print_times != 0 and (cb_params.cur_step_num - self._last_print_time) >= self._per_print_times:
        #     self._last_print_time = cb_params.cur_step_num
            # print("epoch: %s step: %s, loss is %s" % (cb_params.cur_epoch_num, cur_step_in_epoch, loss), flush=True)
        if self.optimizer.dynamic_lr:
            lr = self.optimizer.learning_rate(Tensor(self.optimizer.global_step)).asnumpy()
        else:
            lr = self.optimizer.get_lr().asnumpy()
        if hasattr(cb_params.network, 'loss_scale'):
            loss_scale = getattr(cb_params.network, 'loss_scale', 1) * 1
            loss_scale = loss_scale.asnumpy()
        else:
            loss_scale = 1
        if hasattr(cb_params.train_dataset, 'batch_size'):
            batch_size = getattr(cb_params.train_dataset, 'batch_size', 1)
        else:
            batch_size = 1
        self.trainer_state.update_step(loss, batch_size, loss_scale, lr=lr)
        cb_params = run_context.original_args()

        if cb_params.cur_step_num % self.report_steps == 0:
            self.trainer_state.report_state()
            if self.summary_record is not None:
                self.summary_record.add_value('scalar', 'loss', ms.Tensor(loss))
                self.summary_record.add_value('scalar', 'learning rate', ms.Tensor(lr))
                self.summary_record.add_value('scalar', 'loss_scale', ms.Tensor(loss_scale))
                self.summary_record.record(cb_params.cur_step_num)
        if cb_params.cur_step_num >= self.train_steps:
            run_context.request_stop()

    def __enter__(self):
        # init you summary record in here, when the train script run, it will be inited before training
        if self.summary_dir:
            self.summary_record = ms.train.summary.SummaryRecord(self.summary_dir)
        return self

    def __exit__(self, *exc_args):
        # Note: you must close the summary record, it will release the process pool resource
        # else your training script will not exit from training.
        if self.summary_record:
            self.summary_record.close()


class ModelCheckpointWithBest(ModelCheckpoint):
    def __init__(self, trainer_state, prefix='CKP', directory=None, config=None, load_checkpoint_path=None):
        super().__init__(prefix=prefix, directory=directory, config=config)
        self.trainer_state = trainer_state
        self.best_metric = self.trainer_state.best_metric
        self.load_checkpoint_path = load_checkpoint_path

    def save_best(self, cb_params):
        if self.trainer_state.best_metric > self.best_metric:
            self.best_metric = self.trainer_state.best_metric
            ckpt_path = os.path.join(self._directory, 'best.ckpt')
            if os.path.exists(ckpt_path):
                os.remove(ckpt_path)
            self._save(cb_params, ckpt_path)

    def _save_ckpt(self, cb_params, force_to_save=False):
        """Save checkpoint files."""
        if cb_params.cur_step_num == self._last_triggered_step:
            return

        # if param is cache enable, flush data from cache to host before save_ckpt
        if self._need_flush_from_cache:
            self._flush_from_cache(cb_params)

        save_ckpt = self._check_save_ckpt(cb_params, force_to_save)
        step_num_in_epoch = int((cb_params.cur_step_num - 1) % cb_params.batch_num + 1)

        if save_ckpt:
            cur_ckpoint_file = self._prefix + "-" + str(cb_params.cur_step_num) + "_" \
                + str(cb_params.cur_epoch_num) + ".ckpt"
            # update checkpoint file list.
            self._manager.update_ckpoint_filelist(self._directory, self._prefix)
            # keep checkpoint files number equal max number.
            if self._config.keep_checkpoint_max and 0 < self._config.keep_checkpoint_max <= self._manager.ckpoint_num:
                self._manager.remove_oldest_ckpoint_file()
            elif self._config.keep_checkpoint_per_n_minutes and self._config.keep_checkpoint_per_n_minutes > 0:
                self._cur_time_for_keep = time.time()
                if (self._cur_time_for_keep - self._last_time_for_keep) \
                        < self._config.keep_checkpoint_per_n_minutes * 60:
                    self._manager.keep_one_ckpoint_per_minutes(self._config.keep_checkpoint_per_n_minutes,
                                                               self._cur_time_for_keep)

            # generate the new checkpoint file and rename it.
            # global _save_dir
            # _save_dir = self._directory
            self._save(cb_params, cur_ckpoint_file)
            self.save_best(cb_params)

    def _save(self, cb_params, cur_ckpoint_file):
        cur_file = os.path.join(self._directory, cur_ckpoint_file)
        self._last_time_for_keep = time.time()
        self._last_triggered_step = cb_params.cur_step_num

        if context.get_context("enable_ge"):
            _set_cur_net(cb_params.train_network)
            cb_params.train_network.exec_checkpoint_graph()
        if "epoch_num" in self._append_dict:
            self._append_dict["epoch_num"] = self._append_epoch_num + cb_params.cur_epoch_num
        if "step_num" in self._append_dict:
            self._append_dict["step_num"] = self._append_step_num + cb_params.cur_step_num
        network = self._config.saved_network if self._config.saved_network is not None else cb_params.train_network
        save_checkpoint(network, cur_file, self._config.integrated_save, self._config.async_save,
                        self._append_dict, self._config.enc_key, self._config.enc_mode)

        self._latest_ckpt_file_name = cur_file


class EvalResultsCallback(Callback):
    """
        Get results of evaluation steps
        the result is a Tuple of (Loss, net_outputs, label)
    """
    def __init__(self, result_fn='argmax') -> None:
        super().__init__()
        self.results = {}
        result_fns = {
            'argmax' : ms.ops.Argmax()
        }
        self.result_fn = result_fns.get(result_fn, None)
        parallel_mode = context.get_auto_parallel_context("parallel_mode")
        self.reduce_flag = False
        if parallel_mode in [ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL]:
            self.reduce_flag = True
            # self.allreduce = P.AllReduce()
            self.allgather = AllGather()
        else:
            self.allgather = None

    def step_end(self, run_context):
        loss, net_output, label = run_context.original_args().net_outputs
        # if self.allgather is not None:
        #     label = self.allgather(label)
        if self.result_fn is not None:
            net_output = self.result_fn(net_output)
        loss = loss.view((1, -1))
        loss = loss.asnumpy()
        net_output = net_output.asnumpy()
        label = label.asnumpy()
        if len(self.results) == 0:
            self.results['loss'] = loss
            self.results['net_output'] = net_output
            self.results['label'] = label
        else:
            self.results['loss'] = np.concatenate([self.results['loss'], loss], axis=0)
            self.results['net_output'] = np.concatenate([self.results['net_output'], net_output], axis=0)
            self.results['label'] = np.concatenate([self.results['label'], label], axis=0)
