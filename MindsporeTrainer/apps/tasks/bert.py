# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# zbo@zju.edu.cn
# 2022-08-08
# ============================================================================

from collections import OrderedDict

import numpy as np
import os
import random
import math
from bisect import bisect
import ujson as json
from shutil import copyfile
# from transformers import AutoModelForMultipleChoice

from loguru import logger

from MindsporeTrainer.data import ExampleInstance, ExampleSet, _truncate_segments
from MindsporeTrainer.data.example import *
from MindsporeTrainer.task import EvalData, Task
from MindsporeTrainer.task  import register_task
from utils.metrics import *

import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.dataset as ds
import mindspore.dataset.vision as vision
import mindspore.dataset.transforms as transforms
from mindspore.communication import init, get_rank, get_group_size
from mindspore.nn import Momentum, SoftmaxCrossEntropyWithLogits
import mindspore as ms

from MindsporeTrainer.modeling.modeling_adapter import NetworkWithLoss
from MindsporeTrainer.utils.metrics import BertMetric
from MindsporeTrainer.utils.masker import NGramMaskGenerator
from MindsporeTrainer.data.dynamic_dataset import create_dynamic_dataset
from MindsporeTrainer.modeling.tokenizers import BertTokenizer


@register_task(name="BERT", desc="Basic BERT task")
class BERTTask(Task):
  def __init__(self, data_dir, args=None, **kwargs):
    super().__init__(args, **kwargs)
    self.data_dir = data_dir
    self.args = args
    # self.model_class = resnet50(32, 100)
    self.metric = 'bert'
    self.main_metric = 'perplexity'
    self.optimizer_name = 'Lamb'
    self.tokenizer = BertTokenizer(self.args.vocab_path)
    if args.distribute:
      self.rank_id = get_rank()
      self.rank_size = get_group_size()
    else:
      self.rank_id = 0
      self.rank_size = 1

  def train_data(self, max_seq_len=512, batch_size=32, **kwargs):
    data_path = os.path.join(self.data_dir, 'daizhige.pkl')
    data = self.load_data(data_path, 'GW', max_seq_len)
    # data = ExampleSet(data)
    output_columns = ["input_ids", "input_mask", "token_type_id", "next_sentence_labels",
                  "masked_lm_positions", "masked_lm_ids", "masked_lm_weights"]
    return create_dynamic_dataset(data, self.get_feature_fn(), batch_size=batch_size,
                                  output_columns=output_columns, repeat=self.args.num_train_epochs,
                                  num_workers=self.args.data_workers,num_shards=self.rank_size, shard_id=self.rank_id)

  def eval_data(self, max_seq_len=512, batch_size=32, **kwargs):
    data_path = os.path.join(self.data_dir, 'eval.txt')
    data = self.load_txt_data(data_path, 'GW', max_seq_len)

    output_columns = ["input_ids", "input_mask", "token_type_id", "next_sentence_labels",
                  "masked_lm_positions", "masked_lm_ids", "masked_lm_weights"]
    return create_dynamic_dataset(data, self.get_feature_fn(), batch_size=batch_size,
                                  output_columns=output_columns, repeat=1,
                                  num_workers=self.args.data_workers, num_shards=self.rank_size, 
                                  shard_id=self.rank_id)

  def test_data(self, max_seq_len=512, batch_size=32, **kwargs):
    data_path = os.path.join(self.data_dir, 'daizhige.pkl')
    return self.create_dataset(data_path=data_path, max_seq_len=max_seq_len, batch_size=batch_size)

  def _data(self, name, path, type_name = 'dev', ignore_metric=False, max_examples=None, shuffle=False, max_seq_len=512):
    input_src = os.path.join(self.data_dir, path)
    assert os.path.exists(input_src), f"{input_src} doesn't exists"
    data = self.create_dataset(name)

    predict_fn = self.get_predict_fn(data)
    metrics_fn = self.get_metrics()
    return EvalData(name, data,
      metrics_fn = metrics_fn, predict_fn = predict_fn, ignore_metric=ignore_metric, critial_metrics=['accuracy'])

  def get_metrics(self, **kwargs):
    """Calcuate metrics based on prediction results"""
    return OrderedDict(
      bert = BertMetric(self.args.eval_batch_size),
      )

  # def get_predict_fn(self, data):
  #   """Calcuate metrics based on prediction results"""
  #   def predict_fn(logits, output_dir, name, prefix, targets=None):
  #     output=os.path.join(output_dir, 'submit-{}-{}.tsv'.format(name, prefix))
  #     preds = np.argmax(logits, axis=-1)
  #     labels = self.get_labels()
  #     with open(output, 'w', encoding='utf-8') as fs:
  #       fs.write('targets\tpredictions\n')
  #       if targets is not None:
  #         for i,(e, p) in enumerate(zip(targets, preds)):
  #           fs.write(f'{labels[e]}\t{labels[p]}\n')
  #       else:
  #         for i,(e,p) in enumerate(zip(data, preds)):
  #           fs.write(f'{labels[e.label]}\t{labels[p]}\n')
  #   return predict_fn

  def get_eval_fn(self, **kwargs):
    data = kwargs.get('data')
    if data is None:
      data = self.eval_data(**kwargs)
    def run_eval(model, name, prefix):
      '''
      args: 
        model: Model instance
        name: evaluate name
        prefix: prefix of file
      return:
        float, main metric of this task, used to save best metric model
      '''
      # from wywLM.utils.callbacks import EvalResultsCallback, LossMoniter
      # rc = EvalResultsCallback(result_fn='argmax')
      res = model.eval(data, dataset_sink_mode=False)
      res = res['bert']
      main_metric = res[self.main_metric]
      if self.rank_id == 0:
        output=os.path.join(self.args.output_dir, 'submit-{}-{}.tsv'.format(name, prefix))
        metric_str = '\n'.join([f'{k}: {v:.4f}' for k, v in res.items()])
        metric_str = metric_str + '\n'
        logger.info("====================================")
        logger.info("evaluate result:\n")
        logger.info(metric_str)
        logger.info("====================================")

        with open(output, 'w', encoding='utf-8') as fs:
          fs.write(f"metrics:\n{metric_str}\n")
      return main_metric
    return run_eval


  def get_feature_fn(self, max_seq_len=512, ext_params=None, rng=None, **kwargs):
    tokenizer = self.tokenizer
    mask_generator = NGramMaskGenerator(tokenizer)
    def example_to_feature(*example):
      '''
      sample: text, label
      return: ["input_ids", "input_mask", "token_type_id", "next_sentence_labels",
                  masked_lm_positions", "masked_lm_ids", "masked_lm_weights"]
      '''
      r = rng
      if r is None:
        r = random
      max_num_tokens = max_seq_len - 2
      segments = example[0].tolist().decode('utf-8')
      _tokens = ['[CLS]'] + tokenizer.tokenize(segments) + ['[SEP]']
      if mask_generator:
          tokens, lm_labels = mask_generator.mask_tokens(_tokens, r)
      token_ids = tokenizer.convert_tokens_to_ids(tokens)
      pad_id = tokenizer._convert_token_to_id(tokenizer.pad_token)
      seq_len = len(token_ids)
      pad_num = max_seq_len - seq_len
      token_ids = token_ids + [pad_id] * pad_num
      input_mask = [1] * seq_len + [0] * pad_num
      token_type_id = [0] * max_seq_len
      next_sentence_labels = -1
      masked_lm_positions = [i for i, lbl in enumerate(lm_labels) if lbl != 0]
      masked_lm_ids = [lbl for i, lbl in enumerate(lm_labels) if lbl != 0]
      if len(masked_lm_ids) < mask_generator.max_preds_per_seq:
        masked_lm_positions += [0] * (mask_generator.max_preds_per_seq - len(masked_lm_ids))
        masked_lm_ids += [-1] * (mask_generator.max_preds_per_seq - len(masked_lm_ids))
      masked_lm_weights = [1] * len(masked_lm_ids)
      features = (token_ids,
                  input_mask,
                  token_type_id,
                  next_sentence_labels,
                  masked_lm_positions,
                  masked_lm_ids,
                  masked_lm_weights)
      type_cast_op = transforms.c_transforms.TypeCast(mstype.int32)
      features = [np.array(f, dtype=np.int32) for f in features]
      return tuple(features)

    return example_to_feature

  def get_labels(self):
    """See base class."""
    return [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  
  def load_txt_data(self, path, type=None, max_seq_len=512):
    examples = []
    with open(path, encoding='utf-8') as fs:
        current_chunk = ''
        current_length = 0
        for l in fs:
          l = l.replace('\t', '')
          if self.args.debug:
            if len(examples) >= 3000:
              return examples
          if type == 'JW':
            if random.random() > 0.5:
              continue
          if '\n' in l:
            l_list = l.split('\n')
          else:
            l_list = [l]
          for l in l_list:
            if len(l) == 0:
              continue
            current_chunk += l
            if len(current_chunk) >= max_seq_len:
              example = ExampleInstance(segments=[current_chunk[: max_seq_len - 2]])
              if len(example.segments[0]) < 510:
                print(len(example.segments[0]))
              # examples.append(np.array([current_chunk[: max_seq_len - 2]]))
              # example = np.array([current_chunk[: max_seq_len - 2]])
              examples.append(example)
              current_chunk = current_chunk[max_seq_len - 2: ]
    return ExampleSet(examples)

  def load_data(self, path, type=None, max_seq_len=512):
    import pickle
    examples = []
    with open(path, 'rb') as fs:
      data = pickle.load(fs)
    for cat, files in data.items():
      for path, lines in files.items():
        current_chunk = ''
        current_length = 0
        for l in lines:
          if self.args.debug:
            if len(examples) >= 30000:
              return ExampleSet(examples)
          # if '\n' in l:
          #   l_list = l.split('\n')
          # else:
          #   l_list = [l]
          # for l in l_list:
          if len(l) == 0:
            continue
          current_chunk += l
          if len(current_chunk) >= max_seq_len:
            if current_chunk[0] in ['O', '。', '？', '！', '，', '、', '；', '：', '|']:
              current_chunk = current_chunk[1:]
            example = ExampleInstance(segments=[current_chunk[: max_seq_len - 2]])
            if len(example.segments[0]) < 510:
              print(len(example.segments[0]))
            # examples.append(np.array([current_chunk[: max_seq_len - 2]]))
            examples.append(example)
            current_chunk = current_chunk[max_seq_len - 2: ]
    return ExampleSet(examples)


  def get_model(self):
    from MindsporeTrainer.modeling.models import BertPreTraining
    from MindsporeTrainer.modeling import build_transformer_model
    if self.args.fp16:
      compute_type = mstype.float16
    else:
      compute_type = mstype.float32
    model, config = build_transformer_model(self.args.model_config, 
                                            model=BertPreTraining, 
                                            compute_type=compute_type, 
                                            padding_idx=self.tokenizer._convert_token_to_id(self.tokenizer.pad_token))
    with open(os.path.join(self.args.output_dir, 'config.json'), 'w', encoding='utf-8') as f:
      f.write(config.to_json_string())
    copyfile(self.args.vocab_path, os.path.join(self.args.output_dir, 'vocab.txt'))
    return model
    # return partial_class

  def get_loss(self, *args, **kwargs):
    from MindsporeTrainer.modeling.loss import BertPretrainingLoss
    return BertPretrainingLoss(self.tokenizer.vocab_size)

  def get_eval_head(self, *args, **kwargs):
    from MindsporeTrainer.modeling.models import BertEvalHead
    return BertEvalHead(self.tokenizer.vocab_size)

  def get_opt_fn(self, *args, **kwargs):
    # def opt_fn(args, model, **kwargs):
    #   return Momentum(filter(lambda x: x.requires_grad, model.get_parameters()), 0.01, 0.9)
    return None


