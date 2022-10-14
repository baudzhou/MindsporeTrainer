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
from mindspore.train.model import Model, RunContext, _transfer_tensor_to_tuple, _is_role_pserver,\
                                  _save_final_ckpt, Validator, _InternalCallbackParam, context, _StepSync, \
                                  _enable_distributed_mindrt, _is_role_sched, _cache_enable, _CallbackManager, \
                                  DatasetHelper, connect_network_with_dataset, _set_training_dataset, ParallelMode
from mindspore.common import set_seed


__all__ = ['DistributedTrainer', 'set_random_seed']

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    set_seed(seed)


def requires_grad(model, flag=True):
    for p in model.get_parameters():
        p.requires_grad = flag


class TrainerModel(Model):
    def __init__(self, network, loss_fn=None, optimizer=None, metrics=None, eval_network=None, eval_indexes=None, amp_level="O0", boost_level="O0",
                 gen_update_steps=1, dis_update_steps=1,**kwargs):
        self.generator = network[0]
        self.discriminator = network[1]
        self.gen_update_steps = gen_update_steps
        self.dis_update_steps = dis_update_steps
        if optimizer is not None:
            self.gen_optimizer = optimizer[0]
            self.dis_optimizer = optimizer[1]
        super().__init__(network, loss_fn, optimizer, metrics, eval_network, eval_indexes, amp_level, boost_level, **kwargs)


    def _train_process(self, epoch, train_dataset, list_callback=None, cb_params=None):
        """
        Training process. The data would be passed to network directly.

        Args:
            epoch (int): Total number of iterations on the data.
            train_dataset (Dataset): A training dataset iterator. If there is no
                                     loss_fn, a tuple with multiple data (data1, data2, data3, ...) should be
                                     returned and passed to the network. Otherwise, a tuple (data, label) should
                                     be returned. The data and label would be passed to the network and loss
                                     function respectively.
            list_callback (Callback): Executor of callback list. Default: None.
            cb_params (_InternalCallbackParam): Callback parameters. Default: None.
        """
        dataset_helper, _ = self._exec_preprocess(is_train=True,
                                                  dataset=train_dataset,
                                                  dataset_sink_mode=False,
                                                  epoch_num=epoch)
        cb_params.cur_step_num = 0
        cb_params.dataset_sink_mode = False
        run_context = RunContext(cb_params)
        list_callback.begin(run_context)
        # used to stop training for early stop, such as stopAtTIme or stopATStep
        should_stop = False
        for i in range(epoch):
            cb_params.cur_epoch_num = i + 1

            list_callback.epoch_begin(run_context)

            for next_element in dataset_helper:
                len_element = len(next_element)
                next_element = _transfer_tensor_to_tuple(next_element)
                if self._loss_fn and len_element != 2:
                    raise ValueError("When 'loss_fn' is not None, 'train_dataset' should return "
                                     "two elements, but got {}, please check the number of elements "
                                     "returned by 'train_dataset'".format(len_element))
                cb_params.cur_step_num += 1

                cb_params.train_dataset_element = next_element
                list_callback.step_begin(run_context)
                gen_pred = None
                if (cb_params.cur_step_num + 1) % (self.gen_update_steps) == 0:
                    requires_grad(self.generator.net.backbone.generator, True)
                    requires_grad(self.generator.net.backbone.discriminator, False)
                    # self.generator.net.backbone.generator.set_train(True)
                    # self.generator.net.backbone.discriminator.set_train(False)
                    gen_output = self.generator(*next_element)
                    cb_params.gen_output = gen_output
                    cb_params.net_outputs = gen_output
                    cb_params.generator_output_map = self.generator.output_map
                    if self.generator.output_map is None:
                        gen_pred = gen_output[-1]
                    else:
                        for out, name in zip(gen_output, self.generator.output_map):
                            if name == 'gen_pred':
                                gen_pred = out
                                break
                if (cb_params.cur_step_num + 1) % (self.dis_update_steps) == 0:
                    requires_grad(self.generator.net.backbone.generator, False)
                    requires_grad(self.generator.net.backbone.discriminator, True)
                    dis_output = self.discriminator(*(next_element + [gen_pred]))
                    cb_params.dis_output = dis_output
                    cb_params.discriminator_output_map = self.discriminator.output_map
                # outputs = self._train_network(*next_element)
                # cb_params.net_outputs = outputs
                # if self._loss_scale_manager and self._loss_scale_manager.get_drop_overflow_update():
                #     _, overflow, _ = outputs
                #     overflow = np.all(overflow.asnumpy())
                #     self._loss_scale_manager.update_loss_scale(overflow)

                list_callback.step_end(run_context)
                if _is_role_pserver():
                    os._exit(0)
                should_stop = should_stop or run_context.get_stop_requested()
                if should_stop:
                    break

            train_dataset.reset()

            # if param is cache enable, flush data from cache to host before epoch end
            self._flush_from_cache(cb_params)

            list_callback.epoch_end(run_context)
            should_stop = should_stop or run_context.get_stop_requested()
            if should_stop:
                break

        list_callback.end(run_context)

    def _build_train_network(self):
        self._network = self.generator
        self.generator = super()._build_train_network()
        self._network = self.discriminator
        self.discriminator = super()._build_train_network()


    @_save_final_ckpt
    def _train(self, epoch, train_dataset, callbacks=None, dataset_sink_mode=True, sink_size=-1, initial_epoch=0,
               valid_dataset=None, valid_frequency=1, valid_dataset_sink_mode=True):
        """
        Training.

        Args:
            epoch (int): Total number of iterations on the data.
            train_dataset (Dataset): A training dataset iterator. If there is no
                                     loss_fn, a tuple with multiple data (data1, data2, data3, ...) will be
                                     returned and passed to the network. Otherwise, a tuple (data, label) will
                                     be returned. The data and label would be passed to the network and loss
                                     function respectively.
            callbacks (list): List of callback objects which should be executed while training. Default: None.
            dataset_sink_mode (bool): Determine whether the data should be passed through the dataset channel.
                                      Default: True.
                                      Configure pynative mode or CPU, the training process will be performed with
                                      dataset not sink.
            sink_size (int): Control the amount of data in each sink. Default: -1.
            initial_epoch (int): Epoch at which to start train, it used for resuming a previous training run.
                                 Default: 0.
        """
        epoch = Validator.check_positive_int(epoch)
        if self._parameter_broadcast:
            self.generator.set_broadcast_flag()
            self.discriminator.set_broadcast_flag()

        cb_params = _InternalCallbackParam()
        cb_params.generator = self.generator
        cb_params.discriminator = self.discriminator
        cb_params.train_network = self.generator

        cb_params.epoch_num = epoch - initial_epoch
        if dataset_sink_mode and sink_size > 0:
            cb_params.batch_num = sink_size
        else:
            cb_params.batch_num = train_dataset.get_dataset_size()
        cb_params.mode = "train"
        # cb_params.gen_loss_fn = self.gen_loss_fn
        # cb_params.dis_loss_fn = self.dis_loss_fn
        # cb_params.gen_optimizer = self.gen_optimizer
        # cb_params.dis_optimizer = self.dis_optimizer
        cb_params.parallel_mode = self._parallel_mode
        cb_params.device_number = self._device_number
        cb_params.train_dataset = train_dataset
        cb_params.list_callback = self._transform_callbacks(callbacks)
        valid_infos = (valid_dataset, valid_frequency, valid_dataset_sink_mode)
        if context.get_context("mode") == context.PYNATIVE_MODE:
            cb_params.list_callback.insert(0, _StepSync())
            callbacks = cb_params.list_callback
        cb_params.train_dataset_element = None
        cb_params.network = self._network
        if (_is_role_pserver() and not _enable_distributed_mindrt()) or _is_role_sched():
            epoch = 1
        # Embedding cache server only run one step.
        if (_is_role_pserver() or _is_role_sched()) and _cache_enable():
            epoch = 1
        cb_params.last_save_ckpt_step = None
        cb_params.latest_ckpt_file = None

        # build callback list
        with _CallbackManager(callbacks) as list_callback:
            self._check_reuse_dataset(train_dataset)
            if not dataset_sink_mode:
                self._train_process(epoch, train_dataset, list_callback, cb_params) #, initial_epoch, valid_infos
            elif context.get_context("device_target") == "CPU":
                logger.info("The CPU cannot support dataset sink mode currently."
                            "So the training process will be performed with dataset not sink.")
                self._train_process(epoch, train_dataset, list_callback, cb_params, initial_epoch, valid_infos)
            else:
                self._train_dataset_sink_process(epoch, train_dataset, list_callback,
                                                 cb_params, sink_size, initial_epoch, valid_infos)

    def _exec_preprocess(self, is_train, dataset, dataset_sink_mode, sink_size=-1, epoch_num=1, dataset_helper=None):
        """Initializes dataset."""
        if is_train:
            # network = self._train_network
            phase = 'train'
        else:
            network = self._eval_network
            phase = 'eval'

        if dataset_sink_mode and not is_train:
            dataset.__loop_size__ = 1

        if dataset_helper is None:
            dataset_helper = DatasetHelper(dataset, dataset_sink_mode, sink_size, epoch_num)

        if dataset_sink_mode:
            network = connect_network_with_dataset(network, dataset_helper)

        if is_train:
            _set_training_dataset(dataset_helper)


        self.generator.set_train(is_train)
        self.discriminator.set_train(is_train)
        self.generator.phase = phase
        self.discriminator.phase = phase
        self._backbone_is_train = is_train

        if self._parallel_mode in (ParallelMode.SEMI_AUTO_PARALLEL, ParallelMode.AUTO_PARALLEL):
            self.generator.set_auto_parallel()
            self.discriminator.set_auto_parallel()

        return dataset_helper, None