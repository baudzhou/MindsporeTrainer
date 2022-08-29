# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# zbo@zju.edu.cn
# 2022-08-08
# ============================================================================

import numpy as np

import mindspore.nn as nn
from mindspore.common.initializer import initializer, TruncatedNormal
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.ops import composite as C
from mindspore.common.tensor import Tensor
from mindspore.common.parameter import Parameter
from mindspore.common.api import ms_function
from mindspore.common import dtype as mstype
from mindspore.nn.wrap.grad_reducer import DistributedGradReducer
from mindspore.context import ParallelMode
from mindspore.communication.management import get_group_size
from mindspore.parallel._utils import (_get_device_num, _get_gradients_mean,
                                       _get_parallel_mode, _get_enable_parallel_optimizer)
from mindspore import context
import mindspore as ms

# from wywLM.optims.utils import LossCallBack
from .models import BertModel
from . import msops

GRADIENT_CLIP_TYPE = 1
GRADIENT_CLIP_VALUE = 1.0

clip_grad = C.MultitypeFuncGraph("clip_grad")


@clip_grad.register("Number", "Number", "Tensor")
def _clip_grad(clip_type, clip_value, grad):
    """
    Clip gradients.

    Inputs:
        clip_type (int): The way to clip, 0 for 'value', 1 for 'norm'.
        clip_value (float): Specifies how much to clip.
        grad (tuple[Tensor]): Gradients.

    Outputs:
        tuple[Tensor], clipped gradients.
    """
    if clip_type not in (0, 1):
        return grad
    dt = F.dtype(grad)
    if clip_type == 0:
        new_grad = C.clip_by_value(grad, F.cast(F.tuple_to_array((-clip_value,)), dt),
                                   F.cast(F.tuple_to_array((clip_value,)), dt))
    else:
        new_grad = nn.ClipByNorm()(grad, F.cast(F.tuple_to_array((clip_value,)), dt))
    return new_grad


class NetworkWithLoss(nn.Cell):
    """
    Provide loss through network.

    Args:
        is_training (bool): Specifies whether to use the training mode.
        use_one_hot_embeddings (bool): Specifies whether to use one-hot for embeddings. Default: False.

    Returns:
        Tensor, the loss of the network.
    """

    def __init__(self, backbone, loss_fn, return_all=False, fp16=False):
        super(NetworkWithLoss, self).__init__()
        self.backbone = backbone
        self.loss_fn = loss_fn
        if loss_fn.construct.__code__.co_argcount == 3:
            self.use_simple_loss = True
        else:
            self.use_simple_loss = False
        self.fp16 = fp16
        self.return_all = return_all

    def construct(self, *sample):
        """Get pre-training loss"""
        prediction_scores = self.backbone(*sample)
        if self.use_simple_loss:
            total_loss = self.loss_fn(*(prediction_scores + (sample[1],)))
        else:
            total_loss = self.loss_fn(*(sample + prediction_scores))
        total_loss = F.cast(total_loss, mstype.float32)
        return total_loss


class ModelForEval(nn.Cell):
    '''
    Evaluate scores
    '''
    def __init__(self, network, eval_head=None, fp16=False):
        super(ModelForEval, self).__init__(auto_prefix=False)
        self.eval_head = eval_head
        self.backbone = network.backbone
        if eval_head is None:
            self.eval_head = network.loss_fn
        if eval_head.construct.__code__.co_argcount == 3:
            self.use_simple_eval = True
        else:
            self.use_simple_eval = False
        self.fp16 = fp16
        self.cast = ms.ops.Cast()
        self.reduce_flag = False
        parallel_mode = context.get_auto_parallel_context("parallel_mode")
        if parallel_mode in [ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL] and context.get_context("mode") == context.GRAPH_MODE:
            self.reduce_flag = True
            # self.allreduce = P.AllReduce()
            self.allgather = AllGather() #ms.ops.AllGather()
        else:
            self.allgather = F.identity

    def construct(self, *sample):
        """Calculate prediction scores"""
        pred_scores = self.backbone(*sample)
        if self.use_simple_eval:
            output = self.eval_head(*(pred_scores + (sample[1],)))
        else:
            output = self.eval_head(*(sample + pred_scores))
        out = []
        for m in output:
            if m.ndim == 0:
                m = F.expand_dims(m, 0)
                out.append(self.allgather(m))
            else:
                out.append(self.allgather(m))
        return out


class TrainOneStepCell(nn.TrainOneStepCell):
    """
    Encapsulation class of network training.

    Append an optimizer to the training network after that the construct
    function can be called to create the backward graph.

    Args:
        network (Cell): The training network. Note that loss function should have been added.
        optimizer (Optimizer): Optimizer for updating the weights.
        sens (Number): The adjust parameter. Default: 1.0.
        enable_clip_grad (boolean): If True, clip gradients in BertTrainOneStepCell. Default: True.
    """

    def __init__(self, network, optimizer, sens=1.0, enable_clip_grad=True):
        super(TrainOneStepCell, self).__init__(network, optimizer, sens)
        # self.cast = P.Cast()
        self.hyper_map = C.HyperMap()
        self.enable_clip_grad = enable_clip_grad
        self.enable_tuple_broaden = True

    def set_sens(self, value):
        self.sens = value

    @ms_function
    def clip_grads(self, grads):
        grads = self.hyper_map(F.partial(clip_grad, GRADIENT_CLIP_TYPE, GRADIENT_CLIP_VALUE), grads)
        return grads

    def construct(self, *args):
        """Defines the computation performed."""
        weights = self.weights

        loss = self.network(*args)
        grads = self.grad(self.network, weights)(*args,
                                                 F.cast(F.tuple_to_array((self.sens,)),
                                                           mstype.float32))
        if self.enable_clip_grad:
            grads = self.clip_grads(grads)
        grads = self.grad_reducer(grads)
        self.optimizer(grads)
        return loss


grad_scale = C.MultitypeFuncGraph("grad_scale")
reciprocal = P.Reciprocal()


@grad_scale.register("Tensor", "Tensor")
def tensor_grad_scale(scale, grad):
    return grad * reciprocal(scale)


_grad_overflow = C.MultitypeFuncGraph("_grad_overflow")
grad_overflow = P.FloatStatus()


@_grad_overflow.register("Tensor")
def _tensor_grad_overflow(grad):
    return grad_overflow(grad)


class TrainOneStepWithLossScaleCell(nn.TrainOneStepWithLossScaleCell):
    """
    Encapsulation class of network training.

    Append an optimizer to the training network after that the construct
    function can be called to create the backward graph.

    Args:
        network (Cell): The training network. Note that loss function should have been added.
        optimizer (Optimizer): Optimizer for updating the weights.
        scale_update_cell (Cell): Cell to do the loss scale. Default: None.
    """

    def __init__(self, network, optimizer, scale_update_cell=None, enable_clip_grad=True, opt_overflow=False):
        super(TrainOneStepWithLossScaleCell, self).__init__(network, optimizer, scale_update_cell)
        # self.cast = P.Cast()
        self.degree = 1
        # if self.reducer_flag:
        #     self.degree = get_group_size()
        #     self.grad_reducer = DistributedGradReducer(optimizer.parameters, False, self.degree)
        self.enable_clip_grad = enable_clip_grad
        
        self.loss_scaling_manager = scale_update_cell
        if scale_update_cell:
            self.loss_scale = Parameter(Tensor(scale_update_cell.get_loss_scale(), dtype=mstype.float32))
        else:
            self.loss_scale = Parameter(Tensor(np.array([1]).astype(np.int32)), requires_grad=False)
        self.enable_tuple_broaden = True
        self.opt_overflow = opt_overflow
        self.one = Tensor(np.array([1]).astype(np.int32))
        self.zero = Tensor(np.array([0]).astype(np.int32))

    @ms_function
    def clip_grads(self, grads):
        if self.enable_clip_grad:
            grads = self.hyper_map(F.partial(clip_grad, GRADIENT_CLIP_TYPE, GRADIENT_CLIP_VALUE), grads)
        else:
            grads = F.identity(grads)
        return grads

    def construct(self, *args):
        """Defines the computation performed."""
        weights = self.weights
        loss = self.network(*args)

        # if sens is None:
        #     scaling_sens = self.loss_scale
        # else:
        #     scaling_sens = self.one
        scaling_sens = self.loss_scale
        
        status, scaling_sens = self.start_overflow_check(loss, scaling_sens)
        grad_fn = self.grad(self.network, weights)
        grads = grad_fn(*args, F.cast(scaling_sens, mstype.float32))

        # apply grad reducer on grads
        grads = self.grad_reducer(grads)

        degree_sens = F.cast(scaling_sens * self.degree, mstype.float32)
        grads = self.hyper_map(F.partial(grad_scale, degree_sens), grads)
        
        grads = self.clip_grads(grads)

        cond = self.get_overflow_status(status, grads)
        
        # if sens is None:
        #     overflow = self.loss_scaling_manager(self.loss_scale, cond)
        # else:
        #     overflow = cond
        overflow = self.loss_scaling_manager(self.loss_scale, cond)
        # if self.opt_overflow:
        if not overflow:
            self.optimizer(grads)
        # else:
        #     # if not overflow.asnumpy().tolist():
        #     self.optimizer(grads)

        return (loss, cond, scaling_sens)


class BertTrainOneStepWithLossScaleCell(nn.TrainOneStepWithLossScaleCell):
    """
    Encapsulation class of bert network training.

    Append an optimizer to the training network after that the construct
    function can be called to create the backward graph.

    Args:
        network (Cell): The training network. Note that loss function should have been added.
        optimizer (Optimizer): Optimizer for updating the weights.
        scale_update_cell (Cell): Cell to do the loss scale. Default: None.
    """

    def __init__(self, network, optimizer, scale_update_cell=None):
        super(BertTrainOneStepWithLossScaleCell, self).__init__(network, optimizer, scale_update_cell)
        self.cast = P.Cast()
        self.degree = 1
        if self.reducer_flag:
            self.degree = get_group_size()
            self.grad_reducer = DistributedGradReducer(optimizer.parameters, False, self.degree)

        self.loss_scale = None
        self.loss_scaling_manager = scale_update_cell
        if scale_update_cell:
            self.loss_scale = Parameter(Tensor(scale_update_cell.get_loss_scale(), dtype=mstype.float32))
        self.enable_tuple_broaden = True

    @ms_function
    def clip_grads(self, grads):
        grads = self.hyper_map(F.partial(clip_grad, GRADIENT_CLIP_TYPE, GRADIENT_CLIP_VALUE), grads)
        return grads

    def construct(self,
                  input_ids,
                  input_mask,
                  token_type_id,
                  next_sentence_labels,
                  masked_lm_positions,
                  masked_lm_ids,
                  masked_lm_weights,
                  sens=None):
        """Defines the computation performed."""
        weights = self.weights
        loss = self.network(input_ids,
                            input_mask,
                            token_type_id,
                            next_sentence_labels,
                            masked_lm_positions,
                            masked_lm_ids,
                            masked_lm_weights)
        if sens is None:
            scaling_sens = self.loss_scale
        else:
            scaling_sens = sens
        status, scaling_sens = self.start_overflow_check(loss, scaling_sens)
        grads = self.grad(self.network, weights)(input_ids,
                                                 input_mask,
                                                 token_type_id,
                                                 next_sentence_labels,
                                                 masked_lm_positions,
                                                 masked_lm_ids,
                                                 masked_lm_weights,
                                                 self.cast(scaling_sens,
                                                           mstype.float32))
        # apply grad reducer on grads
        grads = self.grad_reducer(grads)
        degree_sens = self.cast(scaling_sens * self.degree, mstype.float32)
        grads = self.hyper_map(F.partial(grad_scale, degree_sens), grads)
        grads = self.clip_grads(grads)

        cond = self.get_overflow_status(status, grads)
        overflow = cond
        if sens is None:
            overflow = self.loss_scaling_manager(self.loss_scale, cond)
        if not overflow:
            self.optimizer(grads)
        return (loss, cond, scaling_sens)


class TrainOneStepWithLossScaleCellForAdam(nn.TrainOneStepWithLossScaleCell):
    """
    Encapsulation class of network training.

    Append an optimizer to the training network after that the construct
    function can be called to create the backward graph.
    Different from TrainOneStepWithLossScaleCell, the optimizer takes the overflow
    condition as input.

    Args:
        network (Cell): The training network. Note that loss function should have been added.
        optimizer (Optimizer): Optimizer for updating the weights.
        scale_update_cell (Cell): Cell to do the loss scale. Default: None.
    """
    def __init__(self, network, optimizer, scale_update_cell=None, enable_clip_grad=True):
        super(TrainOneStepWithLossScaleCellForAdam, self).__init__(network, optimizer, scale_update_cell)
        self.cast = P.Cast()
        # self.degree = 1
        # if self.reducer_flag:
        #     self.degree = get_group_size()
        #     self.grad_reducer = DistributedGradReducer(optimizer.parameters, False, self.degree)
        self.enable_clip_grad = enable_clip_grad
        self.loss_scale = None
        self.loss_scaling_manager = scale_update_cell
        if scale_update_cell:
            self.loss_scale = Parameter(Tensor(scale_update_cell.get_loss_scale(), dtype=mstype.float32))
        self.enable_tuple_broaden = True

    @ms_function
    def clip_grads(self, grads):
        grads = self.hyper_map(F.partial(clip_grad, GRADIENT_CLIP_TYPE, GRADIENT_CLIP_VALUE), grads)
        return grads

    def construct(self, *args, sens=None):
        """Defines the computation performed."""
        weights = self.weights
        loss = self.network(*args)
        if sens is None:
            scaling_sens = self.loss_scale
        else:
            scaling_sens = sens

        status, scaling_sens = self.start_overflow_check(loss, scaling_sens)
        grads = self.grad(self.network, weights)(*args,
                                                 self.cast(scaling_sens,
                                                           mstype.float32))
        # apply grad reducer on grads
        grads = self.grad_reducer(grads)
        grads = self.hyper_map(F.partial(grad_scale, scaling_sens * self.degree), grads)
        if self.enable_clip_grad:
            grads = self.clip_grads(grads)
        cond = self.get_overflow_status(status, grads)
        overflow = cond
        if self.loss_scaling_manager is not None:
            overflow = self.loss_scaling_manager(scaling_sens, cond)
        self.optimizer(grads, overflow)
        return (loss, cond, scaling_sens)

cast = P.Cast()
add_grads = C.MultitypeFuncGraph("add_grads")


@add_grads.register("Tensor", "Tensor")
def _add_grads(accu_grad, grad):
    return accu_grad + cast(grad, mstype.float32)

update_accu_grads = C.MultitypeFuncGraph("update_accu_grads")

@update_accu_grads.register("Tensor", "Tensor")
def _update_accu_grads(accu_grad, grad):
    succ = True
    return F.depend(succ, F.assign(accu_grad, cast(grad, mstype.float32)))

accumulate_accu_grads = C.MultitypeFuncGraph("accumulate_accu_grads")

@accumulate_accu_grads.register("Tensor", "Tensor")
def _accumulate_accu_grads(accu_grad, grad):
    succ = True
    return F.depend(succ, F.assign_add(accu_grad, cast(grad, mstype.float32)))


zeroslike = P.ZerosLike()
reset_accu_grads = C.MultitypeFuncGraph("reset_accu_grads")


@reset_accu_grads.register("Tensor")
def _reset_accu_grads(accu_grad):
    succ = True
    return F.depend(succ, F.assign(accu_grad, zeroslike(accu_grad)))


class TrainAccumulationAllReducePostWithLossScaleCell(nn.Cell):
    """
    Encapsulation class of network training.

    Append an optimizer to the training network after that the construct
    function can be called to create the backward graph.

    To mimic higher batch size, gradients are accumulated N times before weight update.

    For distribution mode, allreduce will only be implemented in the weight updated step,
    i.e. the sub-step after gradients accumulated N times.

    Args:
        network (Cell): The training network. Note that loss function should have been added.
        optimizer (Optimizer): Optimizer for updating the weights.
        scale_update_cell (Cell): Cell to do the loss scale. Default: None.
        accumulation_steps (int): Number of accumulation steps before gradient update. The global batch size =
                                batch_size * accumulation_steps. Default: 1.
    """

    def __init__(self, network, optimizer, scale_update_cell=None, accumulation_steps=1, 
                    enable_global_norm=False, opt_overflow=False, gpu_target=False):
        super(TrainAccumulationAllReducePostWithLossScaleCell, self).__init__(auto_prefix=False)
        self.network = network
        self.network.set_grad()
        self.weights = optimizer.parameters
        self.optimizer = optimizer
        self.accumulation_steps = accumulation_steps
        self.enable_global_norm = enable_global_norm
        self.one = Tensor(np.array([1]).astype(np.int32))
        self.zero = Tensor(np.array([0]).astype(np.int32))
        self.local_step = Parameter(initializer(0, [1], mstype.int32))
        self.accu_grads = self.weights.clone(prefix="accu_grads", init='zeros')
        self.accu_overflow = Parameter(initializer(0, [1], mstype.int32))
        self.accu_loss = Parameter(initializer(0, [1], mstype.float32))

        self.grad = C.GradOperation(get_by_list=True, sens_param=True)
        self.reducer_flag = False
        self.parallel_mode = context.get_auto_parallel_context("parallel_mode")
        if self.parallel_mode in [ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL]:
            self.reducer_flag = True
        self.grad_reducer = F.identity
        self.degree = 1
        if self.reducer_flag:
            self.degree = _get_device_num()
            self.mean = _get_gradients_mean()
            self.grad_reducer = DistributedGradReducer(optimizer.parameters, self.mean, self.degree)
        self.is_distributed = (self.parallel_mode != ParallelMode.STAND_ALONE)
        self.overflow_reducer = F.identity
        if self.is_distributed:
            self.overflow_reducer = P.AllReduce()
        self.cast = P.Cast()
        self.alloc_status = P.NPUAllocFloatStatus()
        self.get_status = P.NPUGetFloatStatus()
        self.clear_status = P.NPUClearFloatStatus()
        self.reduce_sum = P.ReduceSum(keep_dims=False)
        self.base = Tensor(1, mstype.float32)
        self.less_equal = P.LessEqual()
        self.logical_or = P.LogicalOr()
        self.not_equal = P.NotEqual()
        self.select = P.Select()
        self.reshape = P.Reshape()
        self.hyper_map = C.HyperMap()
        self.loss_scale = None
        self.loss_scaling_manager = scale_update_cell
        self.use_loss_scale = Tensor(self.loss_scaling_manager is not None)
        if scale_update_cell:
            self.loss_scale = Parameter(Tensor(scale_update_cell.get_loss_scale(), dtype=mstype.float32))

        self.opt_overflow = opt_overflow
        self.gpu_target = gpu_target

    def construct(self, *args): #, sens=1
        """Defines the computation performed."""
        # start = time.time()
        weights = self.weights
        
        loss = self.network(*args)
        
        # self.last_weights = [w.asnumpy() for w in weights]
        # if self.not_equal(sens, 1):
        #     scaling_sens = sens
        # else:
        # loss = loss[0]
        if self.use_loss_scale:
            scaling_sens = self.loss_scale
            scaling_sens = C.ones_like(loss) * self.cast(scaling_sens, F.dtype(loss))
            scaling_sens = self.cast(scaling_sens, mstype.float32)
        else:
            scaling_sens = self.one

        # alloc status and clear should be right before gradoperation
        if not self.gpu_target:
            init = self.alloc_status()
            init = F.depend(init, loss)
            clear_status = self.clear_status(init)
            # if self.use_loss_scale:
            scaling_sens = F.depend(scaling_sens, clear_status)
        else:
            init = (False)

        # update accumulation parameters
        is_accu_step = self.not_equal(self.local_step, self.accumulation_steps)
        self.local_step = self.select(is_accu_step, self.local_step + self.one, self.one)
        self.accu_loss = self.select(is_accu_step, self.accu_loss + loss, loss.view((1,)))
        mean_loss = self.accu_loss / self.local_step
        is_accu_step = self.not_equal(self.local_step, self.accumulation_steps)

        grads_fn = self.grad(self.network, weights)
        grads = grads_fn(*args, scaling_sens)

        accu_succ = self.hyper_map(accumulate_accu_grads, self.accu_grads, grads)
        mean_loss = F.depend(mean_loss, accu_succ)
        if not self.gpu_target:
            init = F.depend(init, mean_loss)
            get_status = self.get_status(init)
            init = F.depend(init, get_status)
            flag_sum = self.reduce_sum(init, (0,))
            overflow = self.less_equal(self.base, flag_sum)
        else:
            flag_sum = self.hyper_map(self.partial(_grad_overflow), mean_loss)
            flag_sum = P.AddN()(flag_sum)
            # convert flag_sum to scalar
            flag_sum = P.Reshape()(flag_sum, (()))
            if self.is_distributed:
                flag_reduce = self.allreduce(flag_sum)
                overflow = self.less_equal(self.base, flag_reduce)
            else:
                overflow = self.less_equal(self.base, flag_sum)
        
        overflow = self.logical_or(self.not_equal(self.accu_overflow, self.zero), overflow)
        accu_overflow = self.select(overflow, self.one, self.zero)
        self.accu_overflow = self.select(is_accu_step, accu_overflow, self.zero)

        if is_accu_step:
            # apply grad reducer on grads
            grads = self.grad_reducer(self.accu_grads)
            scaling = scaling_sens * self.degree * self.accumulation_steps
            grads = self.hyper_map(F.partial(grad_scale, scaling), grads)
            # if self.enable_global_norm:
            #     grads = C.clip_by_global_norm(grads, 1.0, None)
            # else:
            #     grads = self.hyper_map(F.partial(clip_grad, GRADIENT_CLIP_TYPE, GRADIENT_CLIP_VALUE), grads)
            accu_overflow = F.depend(accu_overflow, grads)
            accu_overflow = self.overflow_reducer(accu_overflow)
            overflow = self.less_equal(self.base, accu_overflow)
            accu_succ = self.hyper_map(reset_accu_grads, self.accu_grads)
            overflow = F.depend(overflow, accu_succ)
            overflow = self.reshape(overflow, (()))
            # if sens is None:
            if self.use_loss_scale:
                overflow = self.loss_scaling_manager(self.loss_scale, overflow)
                # else:
                #     overflow = False
            if self.opt_overflow:
                # self.last_grads = [g.asnumpy() for g in grads]
                # if np.any(np.isnan(self.last_grads[0])):
                #     print('error')
                self.optimizer(grads, overflow)
            else:
                if not (overflow and self.use_loss_scale):
                    self.optimizer(grads)
        # print(time.time() - start)
        return (mean_loss, overflow, scaling_sens)


class TrainAccumulationAllReduceEachWithLossScaleCell(nn.Cell):
    """
    Encapsulation class of network training.

    Append an optimizer to the training network after that the construct
    function can be called to create the backward graph.

    To mimic higher batch size, gradients are accumulated N times before weight update.

    For distribution mode, allreduce will be implemented after each sub-step and the trailing time
    will be overided by backend optimization pass.

    Args:
        network (Cell): The training network. Note that loss function should have been added.
        optimizer (Optimizer): Optimizer for updating the weights.
        scale_update_cell (Cell): Cell to do the loss scale. Default: None.
        accumulation_steps (int): Number of accumulation steps before gradient update. The global batch size =
                                  batch_size * accumulation_steps. Default: 1.
    """
    def __init__(self, network, optimizer, scale_update_cell=None, accumulation_steps=1, enable_global_norm=False):
        super(TrainAccumulationAllReduceEachWithLossScaleCell, self).__init__(auto_prefix=False)
        self.network = network
        self.network.set_grad()
        self.weights = optimizer.parameters
        self.optimizer = optimizer
        self.accumulation_steps = accumulation_steps
        self.enable_global_norm = enable_global_norm
        self.one = Tensor(np.array([1]).astype(np.int32))
        self.zero = Tensor(np.array([0]).astype(np.int32))
        self.local_step = Parameter(initializer(0, [1], mstype.int32))
        self.accu_grads = self.weights.clone(prefix="accu_grads", init='zeros')
        self.accu_overflow = Parameter(initializer(0, [1], mstype.int32))
        self.accu_loss = Parameter(initializer(0, [1], mstype.float32))

        self.grad = C.GradOperation(get_by_list=True, sens_param=True)
        self.reducer_flag = False
        self.parallel_mode = context.get_auto_parallel_context("parallel_mode")
        if self.parallel_mode in [ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL]:
            self.reducer_flag = True
        self.grad_reducer = F.identity
        self.degree = 1
        if self.reducer_flag:
            self.degree = get_group_size()
            self.grad_reducer = DistributedGradReducer(optimizer.parameters, False, self.degree)
        self.is_distributed = (self.parallel_mode != ParallelMode.STAND_ALONE)
        self.overflow_reducer = F.identity
        if self.is_distributed:
            self.overflow_reducer = P.AllReduce()
        self.cast = P.Cast()
        self.alloc_status = P.NPUAllocFloatStatus()
        self.get_status = P.NPUGetFloatStatus()
        self.clear_before_grad = P.NPUClearFloatStatus()
        self.reduce_sum = P.ReduceSum(keep_dims=False)
        self.base = Tensor(1, mstype.float32)
        self.less_equal = P.LessEqual()
        self.logical_or = P.LogicalOr()
        self.not_equal = P.NotEqual()
        self.select = P.Select()
        self.reshape = P.Reshape()
        self.hyper_map = C.HyperMap()
        self.loss_scale = None
        self.loss_scaling_manager = scale_update_cell
        if scale_update_cell:
            self.loss_scale = Parameter(Tensor(scale_update_cell.get_loss_scale(), dtype=mstype.float32))

    @C.add_flags(has_effect=True)
    def construct(self,
                  *args,
                  sens=None):
        """Defines the computation performed."""
        weights = self.weights
        loss = self.network(*args)[0]
        if sens is None:
            scaling_sens = self.loss_scale
        else:
            scaling_sens = sens

        # update accumulation parameters
        is_accu_step = self.not_equal(self.local_step, self.accumulation_steps)
        self.local_step = self.select(is_accu_step, self.local_step + self.one, self.one)
        self.accu_loss = self.select(is_accu_step, self.accu_loss + loss, loss)
        mean_loss = self.accu_loss / self.local_step
        is_accu_step = self.not_equal(self.local_step, self.accumulation_steps)

        # alloc status and clear should be right before gradoperation
        init = self.alloc_status()
        self.clear_before_grad(init)
        grads = self.grad(self.network, weights)(*args,
                                                 self.cast(scaling_sens,
                                                           mstype.float32))


        accu_grads = self.hyper_map(add_grads, self.accu_grads, grads)
        scaling = scaling_sens * self.degree * self.accumulation_steps
        grads = self.hyper_map(F.partial(grad_scale, scaling), accu_grads)
        grads = self.grad_reducer(grads)

        self.get_status(init)
        flag_sum = self.reduce_sum(init, (0,))
        flag_reduce = self.overflow_reducer(flag_sum)
        overflow = self.less_equal(self.base, flag_reduce)
        overflow = self.logical_or(self.not_equal(self.accu_overflow, self.zero), overflow)
        accu_overflow = self.select(overflow, self.one, self.zero)
        self.accu_overflow = self.select(is_accu_step, accu_overflow, self.zero)
        overflow = self.reshape(overflow, (()))

        if is_accu_step:
            succ = False
            accu_succ = self.hyper_map(update_accu_grads, self.accu_grads, accu_grads)
            succ = F.depend(succ, accu_succ)
        else:
            if sens is None:
                overflow = self.loss_scaling_manager(self.loss_scale, overflow)
            if overflow:
                succ = False
            else:
                if self.enable_global_norm:
                    grads = C.clip_by_global_norm(grads, 1.0, None)
                else:
                    grads = self.hyper_map(F.partial(clip_grad, GRADIENT_CLIP_TYPE, GRADIENT_CLIP_VALUE), grads)

                succ = self.optimizer(grads)

            accu_succ = self.hyper_map(reset_accu_grads, self.accu_grads)
            succ = F.depend(succ, accu_succ)

        ret = (mean_loss, overflow, scaling_sens)
        return F.depend(ret, succ)


class NetworkMatchBucket(nn.Cell):
    '''
    execute according to different sentence lengths.
    '''
    def __init__(self, network, seq_length, bucket_list=None):
        super(NetworkMatchBucket, self).__init__()
        self.network = network
        if not bucket_list or not isinstance(bucket_list, list):
            bucket_list = [seq_length]
        self.bucket_list = [bucket for bucket in bucket_list if bucket <= seq_length]

        if network.reducer_flag:
            reuse_attr = 'reuse_communication_node'
            if not network.grad_reducer.split_fusion:
                hccl_op = network.grad_reducer.allreduce
                network.grad_reducer.allreduce = hccl_op.add_prim_attr(reuse_attr, getattr(hccl_op, 'fusion'))
            else:
                new_op_list = []
                for hccl_op in network.grad_reducer.op_list:
                    new_op = hccl_op.add_prim_attr(reuse_attr, getattr(hccl_op, 'fusion'))
                    new_op_list.append(new_op)
                network.grad_reducer.op_list = new_op_list

    def construct(self,*args,
                  sentence_flag):
        """Switch network according to sentence length."""
        for bucket in self.bucket_list:
            if sentence_flag == bucket:
                input_ids = input_ids[:, :bucket]
                input_mask = input_mask[:, :bucket]
                token_type_id = token_type_id[:, :bucket]
                loss = self.network(*args)
                return loss

        loss = self.network(*args)
        return loss


class AllGather(nn.Cell):
    def __init__(self):
        super(AllGather, self).__init__()
        self.allgather = ms.ops.AllGather()
    def construct(self, x):
        return self.allgather(x)


def model_adapter(model, optimizer, scale_update_cell=None, accumulation_steps=1, 
                    enable_global_norm=False, opt_overflow=False, gpu_target=False):
    if accumulation_steps <= 1:
        if scale_update_cell is None:
            network = TrainOneStepCell(model, optimizer).set_train()
        else:
            network = TrainOneStepWithLossScaleCell(model, optimizer=optimizer, 
                                                    scale_update_cell=scale_update_cell, 
                                                    opt_overflow=opt_overflow)
    else:
        network = TrainAccumulationAllReducePostWithLossScaleCell(model, optimizer=optimizer,
                                                scale_update_cell=scale_update_cell,
                                                accumulation_steps=accumulation_steps,
                                                enable_global_norm=enable_global_norm,
                                                opt_overflow=opt_overflow,
                                                gpu_target=gpu_target).set_train()
    return network
    