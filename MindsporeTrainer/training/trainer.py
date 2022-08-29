# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# zbo@zju.edu.cn
# 2022-08-08
# ============================================================================

import os
import random
import time
import numpy as np
from collections import defaultdict
from loguru import logger

from mindspore.communication.management import get_rank
from mindspore.train.model import Model
from mindspore.nn.wrap.loss_scale import DynamicLossScaleUpdateCell
from mindspore.train.callback import CheckpointConfig
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.common import set_seed

from MindsporeTrainer.utils import amp
from MindsporeTrainer.utils.metrics import MSAucuracy
from MindsporeTrainer.optims import get_optimizer
from MindsporeTrainer.optims.adam import *
from MindsporeTrainer.modeling.modeling_adapter import *
from MindsporeTrainer.apps.tasks import Task
from MindsporeTrainer.utils.callbacks import EvalCallBack, StateCallback, ModelCheckpointWithBest
from MindsporeTrainer.utils.checkpoint import load_ckpt

__all__ = ['DistributedTrainer', 'set_random_seed']

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    set_seed(seed)

class TrainerState:
    def __init__(self, training_steps, name=None, main_metric=''):
        self.__dict__ = defaultdict(float)
        self.loss = 0.0
        self.examples = 0
        self.steps = 0
        self._last_report_step = 0
        self.epochs = 0
        self.next_batch = 0
        self.num_training_steps = training_steps
        self._last_report_time = time.time()
        self.best_steps = 0
        if main_metric in ['loss', 'perplexity']:
            self.neg_metric = True
        else:
            self.neg_metric = False
        self.best_metric = -1e9
        self.name = name
        self.run_id = None
        self.current_loss = 0
        try:
            self.rank = get_rank()
        except:
            self.rank = 0

    def update_step(self, loss, examples, loss_scale, output=None, **kwargs):
        self.examples += examples
        self.loss += loss
        self.current_loss = loss
        self.steps += 1
        self.next_batch += 1
        if not isinstance(loss_scale, (float, int)):
            self.loss_scale = loss_scale.tolist()
        else:
            self.loss_scale = loss_scale
        self.__dict__.update(**kwargs)
    
    def report_state(self):
        if self.steps <= self._last_report_step or self.rank != 0:
            return

        end = time.time()
        start = self._last_report_time
        if self.name is not None:
            tag = f'[{self.name}]'
        else:
            tag = None
        logger.info('{}[{:0.1f}%][{:0.2f}h] Steps={}, loss={:0.5f}, examples={}, loss_scale={:0.1f}, {:0.4f}s'.format(
            tag, 100*self.steps/self.num_training_steps, \
            (self.num_training_steps - self.steps)*(end - start)/((self.steps-self._last_report_step)*3600), 
            self.steps, self.current_loss, self.examples, self.loss_scale, end-start))
        self._last_report_time = end
        self._last_report_step = self.steps


def get_eval_fn(task, data, output_dir, main_metric='acc', custom_metric_fn=None, rank=0):
    def run_eval(model, name, prefix):
        from MindsporeTrainer.utils.callbacks import EvalResultsCallback, LossMoniter
        rc = EvalResultsCallback(result_fn='argmax')
        res = model.eval(data, callbacks = [rc, LossMoniter(10)], dataset_sink_mode=False)
        if rank == 0:
            result = rc.results
            output=os.path.join(output_dir, 'submit-{}-{}.tsv'.format(name, prefix))
            targets = result.get('label')
            preds = result.get('net_output')

            if custom_metric_fn is not None:
                cms = custom_metric_fn(labels=targets, predicts=preds)
                for m, m_r in cms.items():
                    if m in res:
                        m = m + '_c'
                    res[m] = m_r
            metric_str = '\n'.join([f'{k}: {v:.4f}' for k, v in res.items()])
            metric_str = metric_str + '\n'
            logger.info("====================================")
            logger.info("evaluate result:\n")
            logger.info(metric_str)
            logger.info("====================================")

            labels = task.get_labels()
            with open(output, 'w', encoding='utf-8') as fs:
                fs.write(f"metrics:\n{metric_str}")
                fs.write('targets\tpredictions\n')
                if targets is not None:
                    for i,(e, p) in enumerate(zip(targets, preds)):
                        if p.size > 1:
                            p = ''.join([labels[pp] for pp in p])
                            e = ''.join([labels[pp] for pp in e])
                        else:
                            p = labels[p]
                            e = labels[e]
                        fs.write(f'{e}\t{p}\n')
                else:
                    for i,(e,p) in enumerate(zip(data, preds)):
                        if p.size > 1:
                            p = ''.join([labels[pp] for pp in p])
                            e = ''.join([labels[pp] for pp in e])
                        else:
                            p = labels[p]
                            e = labels[e]
                        fs.write(f'{e}\t{p}\n')
        return res[main_metric]
    return run_eval


def get_pred_fn(task, model, data, output_dir, name, prefix='final', sequence=False, dump=True):
    def run_predict():
        result = model.predict(data, dataset_sink_mode=True)
        if sequence:
            if result.ndim() == 3:
                result = result.argmax(-1)
        else:
            if result.ndim() == 2:
                result = result.argmax(-1)
        result = result.asnumpy()
        if dump:
            output=os.path.join(output_dir, 'submit-{}-{}.tsv'.format(name, prefix))
            print("====================================")
            print(f"predict finished. writing resutls to {output}")
            print("====================================")

            labels = task.get_labels()
            with open(output, 'w', encoding='utf-8') as fs:
                for i, p in enumerate(result):
                    if p.size > 0:
                        p = ''.join([labels[pp] for pp in P])
                    else:
                        p = labels[p]
                    fs.write(f'{p}\n')
        return result
    return run_predict

class DistributedTrainer:
    def __init__(self, args, task: Task, **kwargs):
        """
        data_fn return tuples (training_dataset, training_steps, train_sampler, batch_scheduler), training_dataset is required
        loss_fn return the loss of current mini-batch and the size of the batch
        optimizer_fn return the created optimizer
        eval_fn return metrics for model selection
        """
        self.__dict__.update(kwargs)
        self.args = args
        self.task = task
        self.initialized = None
        self.before_run()

        self.accumulative_update = 1
        if hasattr(args, 'accumulative_update'):
            self.accumulative_update = args.accumulative_update

        self.output_dir = args.output_dir
        self.trainer_state = TrainerState(args.train_steps, name=args.tag, main_metric=self.task.main_metric)
        self.report_interval = 10

    def before_run(self):
        if not self.initialized:
            self.initialize()
        if self.args.num_train_epochs <= 0:
            self.args.num_train_epochs = 1
        self.init_task()
        self._setup_model()
        if self.args.do_train:
            if self.args.train_steps <= 0:
                self.args.train_steps = self.args.num_train_epochs * self.train_data.get_dataset_size() // self.args.accumulation_steps
                logger.info("train steps: {}".format(self.args.train_steps))
            else:
                self.args.train_steps = self.args.train_steps * self.args.accumulation_steps
                self.args.num_train_epochs = self.args.train_steps // self.train_data.get_dataset_size() + 1
            self.args.train_steps = int(self.args.train_steps)
            self.args.num_train_epochs = int(self.args.num_train_epochs)
            self.args.save_eval_steps = int(self.args.save_eval_steps / self.args.device_num)

    def init_task(self):
        label_list = self.task.get_labels()
        self.model = self.task.get_model()
        logger.info(f'Total parameters: {sum([p.size for p in self.model.get_parameters()])}')

        self.test_data = None
        logger.info(f"    Evaluation batch size = {self.args.eval_batch_size}")
        if self.args.do_predict:
            self.test_data = self.task.test_data()
            logger.info(f"    Prediction batch size = {self.args.predict_batch_size}")

        if self.args.do_train:
            self.train_data = self.task.train_data(batch_size=self.args.train_batch_size, debug=self.args.debug)
        else:
            self.train_data = None
        optimizer_fn = self.task.get_opt_fn()
        self.optimizer_fn = optimizer_fn if optimizer_fn is not None else get_optimizer

        def _loss_fn(trainer, model, batch):
            _,loss = model(**batch)
            batch_size = batch['input_ids'].size(0)
            return loss.mean(), batch_size
        loss = self.task.get_loss()
        if loss is None:
            loss = self.task.get_loss_fn(self.model)
        self.loss_fn = loss if loss is not None else _loss_fn
        self.eval_head = self.task.get_eval_head()

    def initialize(self):
        set_random_seed(self.args.seed)
        self.initialized = True

    def run(self):
        callbacks = []
        if self.args.fp16 and self.args.device_target != 'Ascend':
            self.model = amp.build_train_network(self.model, self.loss_fn, level='O2')
        else:
            self.model = NetworkWithLoss(self.model, self.loss_fn, return_all=True)
        if len(self.args.load_checkpoint_path) > 0:
            load_ckpt(self.model, self.args.load_checkpoint_path)
        if self.args.do_eval:
            metrics = self.task.get_metrics()
            self.eval_data = self.task.eval_data()
            if metrics is None:
                metrics = {'acc': MSAucuracy()}
        else:
            self.eval_model = None
            metrics = None

        if self.args.do_train:
            self.optimizer, opt_overflow = self.optimizer_fn(self.args, self.model, opt_name=self.task.optimizer_name)

            if self.args.enable_save_ckpt == "true" and self.args.rank % min(8, self.args.device_num) == 0:
                    config_ck = CheckpointConfig(save_checkpoint_steps=self.args.save_eval_steps,
                                                keep_checkpoint_max=self.args.save_checkpoint_num)
                    ckpoint_cb = ModelCheckpointWithBest(self.trainer_state, prefix=self.args.task_name,
                                                        directory=self.args.output_dir, config=config_ck)
                    callbacks.append(ckpoint_cb)
            self.initialized = False
            if self.args.fp16 and self.args.enable_lossscale == "true":
                    update_cell = DynamicLossScaleUpdateCell(loss_scale_value=self.args.loss_scale_value,
                                                            scale_factor=self.args.scale_factor,
                                                            scale_window=self.args.scale_window)
            else:
                update_cell = None
            accumulation_steps = self.args.accumulation_steps
            enable_global_norm = self.args.enable_global_norm
            network = model_adapter(self.model, optimizer=self.optimizer,
                                    scale_update_cell=update_cell,
                                    accumulation_steps=accumulation_steps,
                                    enable_global_norm=enable_global_norm,
                                    opt_overflow=opt_overflow,
                                    gpu_target=self.args.device_target=='GPU')

            summary_dir = os.path.join(self.output_dir, 'tblogger', str(self.args.rank))
            callbacks.append(StateCallback(self.trainer_state, self.optimizer, 
                                           report_steps=self.args.report_interval,
                                           summary_dir=summary_dir, train_steps=self.args.train_steps))
            if self.args.do_eval:
                self.eval_model = ModelForEval(self.model, self.eval_head, fp16=self.args.fp16)

            model = Model(network, eval_network=self.eval_model, metrics=metrics)
            if self.args.thor:
                from mindspore.train.train_thor import ConvertModelUtils
                model = ConvertModelUtils().convert_to_thor_model(model, network=network, optimizer=self.optimizer)
            if self.args.do_eval:
                main_metric = self.task.main_metric
                eval_fn = self.task.get_eval_fn(data=self.eval_data, output_dir=self.args.output_dir, rank=self.args.rank)
                if eval_fn is None:
                    eval_fn = get_eval_fn(self.task, self.eval_data, self.args.output_dir, 
                                          rank=self.args.rank, main_metric=main_metric)
                eval_callback = EvalCallBack(model, self.trainer_state, self.eval_data, 0, eval_fn, self.args.train_steps,
                                             eval_steps=self.args.save_eval_steps, rank=self.args.local_rank)
                callbacks = [eval_callback] + callbacks

            model.train(self.args.num_train_epochs, 
                        self.train_data, 
                        callbacks=callbacks, 
                        dataset_sink_mode=self.args.enable_data_sink,
                        sink_size=self.args.data_sink_steps
                        )

        elif self.args.do_predict:
            self.model = ModelForEval(self.model.net, self.eval_head)
            if len(self.args.load_checkpoint_path) > 0:
                load_ckpt(self.model, self.args.load_checkpoint_path)
            model = Model(self.model)
            pred_fn = self.task.get_pred_fn(data=self.eval_data, output_dir=self.args.output_dir, rank=self.args.local_rank)
            if pred_fn is None:
                pred_fn = get_pred_fn()
            result = pred_fn(self.task, model, self.test_data, self.args.output_dir, 'test')
        elif self.args.do_eval and not self.args.do_train:
            self.model = ModelForEval(self.model, self.eval_head)
            if len(self.args.load_checkpoint_path) > 0:
                load_ckpt(self.model, self.args.load_checkpoint_path)
            model = Model(self.model, eval_network=self.model, metrics=metrics)
            main_metric = self.task.main_metric
            metric = self.task.metric
            eval_fn = self.task.get_eval_fn(data=self.eval_data, output_dir=self.args.output_dir, rank=self.args.local_rank)
            if eval_fn is None:
                eval_fn = get_eval_fn(self.task, self.eval_data, self.args.output_dir, 
                                      rank=self.args.local_rank, main_metric=main_metric)
            eval_fn(model, 'eval', 'final')

    def _setup_model(self):
        if len(self.args.load_checkpoint_path) > 0:
                param_dict = load_checkpoint(self.args.load_checkpoint_path)
                load_param_into_net(self.model, param_dict)
