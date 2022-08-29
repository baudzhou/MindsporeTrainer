# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# zbo@zju.edu.cn
# 2022-08-08
# ============================================================================

import numpy as np
import math
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from scipy.stats import pearsonr, spearmanr
from statistics import *
from scipy.special import softmax
from mindspore.nn.metrics import Metric, rearrange_inputs
# from mindspore.nn.metrics.metric import _check_onehot_data
from mindspore.nn.metrics.accuracy import Accuracy
import mindspore.ops.functional as F
import mindspore.common.dtype as mstype
from mindspore._checkparam import Validator as validator

def metric_multi_accuracy(logits, labels, options_num):
  logits = np.reshape(softmax(logits, -1)[:,1], (len(logits)//options_num, options_num))
  labels = np.argmax(np.reshape(labels, (len(labels)//options_num, options_num)),-1)
  return metric_accuracy(logits, labels)

def metric_accuracy(logits=None, labels=None, predicts=None):
  if logits is not None:
    predicts = np.argmax(logits, axis=1)
  return accuracy_score(labels, predicts)

def metric_precision(logits=None, labels=None, predicts=None, label_cls=None):
  if logits is not None:
    predicts = np.argmax(logits, axis=1)
  if labels.max() > 1:
    return precision_score(labels, predicts, average='macro', labels=label_cls)
  else:
    return precision_score(labels, predicts)

def metric_recall(logits=None, labels=None, predicts=None, label_cls=None):
  if logits is not None:
    predicts = np.argmax(logits, axis=1)
  if labels.max() > 1:
    return recall_score(labels, predicts, average='macro', labels=label_cls)
  else:
    return recall_score(labels, predicts)

def metric_f1(logits=None, labels=None, predicts=None, label_cls=None):
  if logits is not None:
    predicts = np.argmax(logits, axis=1)
  if labels.max() > 1:
    return f1_score(labels, predicts, average='macro', labels=label_cls)
  else:
    return f1_score(labels, predicts)

def metric_macro_f1(logits, ground_truth, labels=[0,1]):
  if logits is not None:
    predicts = np.argmax(logits, axis=1)
  f1=[]
  for l in labels:
    binary_g = (ground_truth==l).astype(np.int)
    binary_p = (predicts==l).astype(np.int)
    f1.append(f1_score(binary_g, binary_p))
  return float(np.mean(f1))

def metric_mcc(logits=None, labels=None, predicts=None):
  if logits is not None:
    predicts = np.argmax(logits, axis=1)
  return matthews_corrcoef(labels, predicts)


class BertMetric(Metric):
    """
    The metric of bert network.
    Args:
        batch_size (int): The batchsize of each device.
    """
    def __init__(self, batch_size):
        super(BertMetric, self).__init__()
        self.batch_size = batch_size
        self.ppl = Perplexity()
        self.clear()

    def clear(self):
        self.mlm_total = 0
        self.mlm_acc = 0
        self.loss = 0
        self.steps = 0
        self.ppl.clear()

    def update(self, *inputs):
        """Defines the computation performed.
            inputs: ["input_ids", "input_mask", "token_type_id", "next_sentence_labels",
                     "masked_lm_positions", "masked_lm_ids", "masked_lm_weights"] + 
                    [prediction_scores, seq_relationship_score]
        """
        total_loss, acc, total, prediction_scores, masked_lm_ids = inputs
        mlm_acc = self._convert_data(acc)
        mlm_total = self._convert_data(total)
        self.mlm_acc += mlm_acc.sum()
        self.mlm_total += mlm_total.sum()
        self.loss += self._convert_data(total_loss.mean()).squeeze().tolist()
        self.steps += 1
        self.ppl.update(prediction_scores, masked_lm_ids)

    def eval(self):
        return {'acc':self.mlm_acc / self.mlm_total, 
                'loss': self.loss / self.steps, 
                'perplexity': self.ppl.eval()}


class MSAucuracy(Accuracy):
  def __call__(self, *inputs):
    """
    Evaluate input data once.

    Args:
        inputs (tuple): The first item is a predict array, the second item is a target array.

    Returns:
        Float, compute result.
    """
    if len(inputs) == 3:
      new_inputs = inputs[1:]
    self.clear()
    self.update(*new_inputs)
    return self.eval()

  def update(self, *inputs):
      if len(inputs) == 3:
        inputs = inputs[1:]
      if len(inputs) != 2:
          raise ValueError("For 'Accuracy.update', it needs 2 inputs (predicted value, true value), "
                            "but got {}".format(len(inputs)))
      y_pred = self._convert_data(inputs[0])
      y = self._convert_data(inputs[1])
      if self._type == 'classification' and y_pred.ndim != y.ndim:
          y_pred = y_pred.argmax(axis=1)
      # self._check_shape(y_pred, y)
      # self._check_value(y_pred, y)

      # if self._class_num == 0:
      #     self._class_num = y_pred.shape[1]
      # elif y_pred.shape[1] != self._class_num:
      #     raise ValueError("For 'Accuracy.update', class number not match, last input predicted data contain {} "
      #                       "classes, but current predicted data contain {} classes, please check your predicted "
      #                       "value(inputs[0]).".format(self._class_num, y_pred.shape[1]))

      if self._type == 'classification':
        if y_pred.ndim > 1:
          indices = y_pred.argmax(axis=1)
          result = (np.equal(indices, y) * 1).reshape(-1)
        else:
          result = (np.equal(y_pred, y) * 1).reshape(-1)
      elif self._type == 'multilabel':
          dimension_index = y_pred.ndim - 1
          y_pred = y_pred.swapaxes(1, dimension_index).reshape(-1, self._class_num)
          y = y.swapaxes(1, dimension_index).reshape(-1, self._class_num)
          result = np.equal(y_pred, y).all(axis=1) * 1

      self._correct_num += result.sum()
      self._total_num += result.shape[0]



class Perplexity(Metric):
    r"""
    Computes perplexity. Perplexity is a measurement about how well a probability distribution or a model predicts a
    sample. A low perplexity indicates the model can predict the sample well. The function is shown as follows:

    .. math::
        PP(W)=P(w_{1}w_{2}...w_{N})^{-\frac{1}{N}}=\sqrt[N]{\frac{1}{P(w_{1}w_{2}...w_{N})}}

    Args:
        ignore_label (int): Index of an invalid label to be ignored when counting. If set to `None`, it will include all
                            entries. Default: -1.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Note:
        The method `update` must be called with the form `update(preds, labels)`.

    Examples:
        >>> import numpy as np
        >>> from mindspore import nn, Tensor
        >>>
        >>> x = Tensor(np.array([[0.2, 0.5], [0.3, 0.1], [0.9, 0.6]]))
        >>> y = Tensor(np.array([1, 0, 1]))
        >>> metric = nn.Perplexity(ignore_label=None)
        >>> metric.clear()
        >>> metric.update(x, y)
        >>> perplexity = metric.eval()
        >>> print(perplexity)
        2.231443166940565
    """

    def __init__(self, ignore_label=-1):
        super(Perplexity, self).__init__()

        if ignore_label is None:
            self.ignore_label = ignore_label
        else:
            self.ignore_label = validator.check_value_type("ignore_label", ignore_label, [int])
        self.clear()

    def clear(self):
        """Clears the internal evaluation result."""
        self._sum_metric = 0.0
        self._num_inst = 0

    @rearrange_inputs
    def update(self, *inputs):
        """
        Updates the internal evaluation result: math:preds and :math:labels.

        Args:
            inputs: Input `preds` and `labels`. `preds` and `labels` are Tensor, list or numpy.ndarray.
                    `preds` is the predicted values, `labels` is the label of the data.
                    The shape of `preds` and `labels` are both :math:`(N, C)`.

        Raises:
            ValueError: If the number of the inputs is not 2.
            RuntimeError: If preds and labels have different lengths.
            RuntimeError: If label shape is not equal to pred shape.
        """
        if len(inputs) != 2:
            raise ValueError('The perplexity needs 2 inputs (preds, labels), but got {}.'.format(len(inputs)))

        preds = [self._convert_data(inputs[0])]
        labels = [self._convert_data(inputs[1])]

        if len(preds) != len(labels):
            raise RuntimeError('The preds and labels should have the same length, but the length of preds is{}, '
                               'the length of labels is {}.'.format(len(preds), len(labels)))

        loss = 0.
        num = 0
        for label, pred in zip(labels, preds):
          # pred scores are log_softmax, so it should be apply np.exp() to get probability
          pred = np.exp(pred)
          if label.size != pred.size / pred.shape[-1]:
              raise RuntimeError("shape mismatch: label shape should be equal to pred shape, but got label shape "
                                  "is {}, pred shape is {}.".format(label.shape, pred.shape))
          label = label.reshape((label.size,))
          label_expand = label.astype(int)
          label_expand = np.expand_dims(label_expand, axis=1)
          first_indices = np.arange(label_expand.shape[0])[:, None]
          pred = pred.reshape(-1, pred.shape[-1])
          pred = np.squeeze(pred[first_indices, label_expand]).astype(np.float32)
          if self.ignore_label is not None:
              ignore = (label == self.ignore_label).astype(pred.dtype)
              num -= np.sum(ignore)
              pred = pred * (1 - ignore) + ignore
          loss -= np.sum(np.log(np.maximum(1e-10, pred)))
          num += pred.size
        self._sum_metric += loss
        self._num_inst += num

    def eval(self):
        r"""
        Returns the current evaluation result.

        Returns:
            float, the computed result.

        Raises:
            RuntimeError: If the sample size is 0.
        """
        if self._num_inst == 0:
            raise RuntimeError('The perplexity can not be calculated, because the number of samples is 0.')

        return math.exp(self._sum_metric / self._num_inst)
