# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# zbo@zju.edu.cn
# 2022-08-08
# ============================================================================

from collections import OrderedDict

import numpy as np
import os
from shutil import copyfile
from loguru import logger
from MindsporeTrainer.data import ExampleInstance, ExampleSet, _truncate_segments
from MindsporeTrainer.data.example import *
from MindsporeTrainer.task import EvalData, Task
from MindsporeTrainer.task  import register_task
from utils.metrics import *

from mindspore.communication import init, get_rank, get_group_size

from MindsporeTrainer.modeling.modeling_adapter import NetworkWithLoss
from MindsporeTrainer.utils.metrics import BertMetric, MSAucuracy
from MindsporeTrainer.utils.masker import NGramMaskGenerator
from MindsporeTrainer.data.dynamic_dataset import create_dynamic_dataset
from MindsporeTrainer.modeling.tokenizers import BertTokenizer
from MindsporeTrainer.modeling.layers import ClsEvalHead
from ..models.classify import BertForClassify, ClsLoss

@register_task(name="GJC", desc="GJC task")
class GJCTask(Task):
  def __init__(self, data_dir, args=None, **kwargs):
    super().__init__(args, **kwargs)
    self.data_dir = data_dir
    self.args = args
    # self.model_class = resnet50(32, 100)
    self.metric = 'acc'
    self.main_metric = 'acc'
    self.optimizer_name = 'Lamb'
    self.tokenizer = BertTokenizer(self.args.vocab_path)
    self.label_to_id = {l: i for i, l in enumerate(self.get_labels())}
    if args.distribute:
      self.rank_id = get_rank()
      self.rank_size = get_group_size()
    else:
      self.rank_id = 0
      self.rank_size = 1

  def train_data(self, max_seq_len=512, batch_size=32, **kwargs):
    data = self.load_data(os.path.join(self.data_dir, 'train.tsv'), max_seq_len=max_seq_len)
    # data = ExampleSet(data)
    output_columns = ["input_ids", "input_mask", "token_type_id", "labels"]
    return create_dynamic_dataset(data, self.get_feature_fn(), batch_size=batch_size,
                                  output_columns=output_columns, repeat=self.args.num_train_epochs,
                                  num_workers=self.args.data_workers,num_shards=self.rank_size, shard_id=self.rank_id)

  def eval_data(self, max_seq_len=512, batch_size=32, **kwargs):
    data = self.load_data(os.path.join(self.data_dir, 'dev.tsv'), max_seq_len=max_seq_len)
    # data = ExampleSet(data)
    output_columns = ["input_ids", "input_mask", "token_type_id", "labels"]
    return create_dynamic_dataset(data, self.get_feature_fn(), batch_size=batch_size,
                                  output_columns=output_columns, repeat=self.args.num_train_epochs,
                                  num_workers=self.args.data_workers,num_shards=self.rank_size, shard_id=self.rank_id)

  def test_data(self, max_seq_len=512, batch_size=32, **kwargs):
    data = self.load_data(os.path.join(self.data_dir, 'test.tsv'), max_seq_len=max_seq_len)
    # data = ExampleSet(data)
    output_columns = ["input_ids", "input_mask", "token_type_id", "labels"]
    return create_dynamic_dataset(data, self.get_feature_fn(), batch_size=batch_size,
                                  output_columns=output_columns, repeat=self.args.num_train_epochs,
                                  num_workers=self.args.data_workers,num_shards=self.rank_size, shard_id=self.rank_id)

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
    return {'acc': MSAucuracy()}

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
      # res = res['acc']
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
    def example_to_feature(*example):
      # print(example)
      features = OrderedDict()
      text = example[0].tolist().decode('utf-8')
      # text = [t.decode('utf-8') for t in text]
      label = example[1].tolist()[0]
      # print(f'sampel: {len(example)}, {text}, {label}\n')
      tokens = ['[CLS]'] + list(text) + ['[SEP]']
      num_pads = max_seq_len - len(tokens)
      pad_id = tokenizer._convert_token_to_id(tokenizer.pad_token)
      token_ids = tokenizer.convert_tokens_to_ids(tokens) + [pad_id] * num_pads
      input_mask = [1] * len(tokens) + [0] * num_pads
      token_type_id = [0] * max_seq_len
      features = (token_ids,
                  input_mask,
                  token_type_id,
                  label)
      features = [np.array(f, dtype=np.int32) for f in features]
      return tuple(features)
    return example_to_feature

  def get_labels(self):
    """See base class."""
    return ['儒藏', '诗藏', '佛藏', '易藏', '医藏', '艺藏', '子藏', '集藏', '道藏', '史藏']
  
  def load_data(self, path, max_seq_len=512, max_examples=None, shuffle=False):
    max_len = max_seq_len - 2
    examples=[]
    merged_words = []
    merged_tokens = []
    merged_labels = []
    size = 0
    with open(path, 'r', encoding='utf-8') as f:
      for sent in f:
        label, sent = sent.strip().split('\t')
        label = self.label_to_id[label]

        # tokens = self.tokenizer.tokenize(sent) # for w in sent
        examples.append(ExampleInstance(segments=[sent], label=label, sentence=sent))
        if self.args.debug and len(examples) >= 3000:
          break

      def get_stats(l):
        return f'Max={max(l)}, min={min(l)}, avg={np.mean(l)}'
      ctx_token_size = [sum(len(w) for w in  e.segments[0]) for e in examples]
      logger.info(f'Statistics: {get_stats(ctx_token_size)}, \
                    long={len([t for t in ctx_token_size if t > 500])}/{len(ctx_token_size)}')
      # return examples
    return ExampleSet(examples)


  def get_model(self):
    from MindsporeTrainer.modeling import build_transformer_model, BertModel
    if self.args.fp16:
      compute_type = mstype.float16
    else:
      compute_type = mstype.float32
    model, config = build_transformer_model(self.args.model_config, 
                                            model=BertModel, 
                                            compute_type=compute_type, 
                                            padding_idx=self.tokenizer._convert_token_to_id(self.tokenizer.pad_token))
    model = BertForClassify(model, num_labels=len(self.get_labels()))
    with open(os.path.join(self.args.output_dir, 'config.json'), 'w', encoding='utf-8') as f:
      f.write(config.to_json_string())
    copyfile(self.args.vocab_path, os.path.join(self.args.output_dir, 'vocab.txt'))
    return model

  def get_loss(self, *args, **kwargs):
    return ClsLoss(reduction='mean')

  def get_eval_head(self, *args, **kwargs):
    from MindsporeTrainer.modeling.models import BertEvalHead
    return ClsEvalHead(len(self.get_labels()))

  def get_opt_fn(self, *args, **kwargs):
    # def opt_fn(args, model, **kwargs):
    #   return Momentum(filter(lambda x: x.requires_grad, model.get_parameters()), 0.01, 0.9)
    return None


