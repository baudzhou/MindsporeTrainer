# MindsporeTrainer
 ![PyPI](https://img.shields.io/pypi/v/MindsporeTrainer?color=blue) 
 ![PyPI - Python Version](https://img.shields.io/pypi/pyversions/MindsporeTrainer)
 ![GitHub Workflow Status](https://img.shields.io/github/workflow/status/baudzhou/MindsporeTrainer/Upload%20Python%20Package)
  ![issues](https://img.shields.io/github/issues/baudzhou/MindsporeTrainer) 
  ![License](https://img.shields.io/github/license/baudzhou/MindsporeTrainer) 

**MindsporeTrainer** 是基于昇思MindSpore的训练框架。让Mindspore的算法研究更容易一些。  
  
Mindspore上手不易，希望能帮助各位炼丹师的升级之路更容易些。  

[**Home page**](https://github.com/baudzhou/MindsporeTrainer)
# 目录
* [主要特性](#主要特性)
* [安装](#安装)
* [使用方法](#使用方法)
    * [DeBERTa预训练任务示例](#DeBERTa预训练任务示例)
    * [多卡训练](#多卡训练)
* [模型创建](#模型创建)
* [参数介绍](#参数介绍)
* [Task](#Task)
* [API](#API)
* [models](#models)
# 主要特性
主要的几个出发点是：  
+ 采用纯python实现，方便多卡训练过程中的调试
+ 易于扩展，对新任务采用插件式接入
+ 方便实现多种模型的训练、评估、预测等  
# 安装
## pip
`pip install MindsporeTrainer`
## 源码
`python setup.py`
# 使用方法
## DeBERTa预训练任务示例
### 1. 定义task
[MindsporeTrainer/apps/tasks/deberta.py](./MindsporeTrainer/apps/tasks/deberta.py)
```
from collections import OrderedDict
import numpy as np
import os
import random
from shutil import copyfile
from loguru import logger

from mindspore.communication import get_rank, get_group_size

from MindsporeTrainer.data import ExampleInstance, ExampleSet
from MindsporeTrainer.data.example import *
from MindsporeTrainer.apps.tasks import EvalData, TransformerTask
from MindsporeTrainer.apps.tasks import register_task
from MindsporeTrainer.utils.metrics import *
from MindsporeTrainer.utils.metrics import BertMetric
from MindsporeTrainer.utils.masker import NGramMaskGenerator
from MindsporeTrainer.data.dynamic_dataset import create_dynamic_dataset
from MindsporeTrainer.modeling.tokenizers import BertTokenizer

@register_task(name="DEBERTA", desc="Basic DEBERTA task")
class DEBERTATask(TransformerTask):
    def __init__(self, data_dir, args, **kwargs):
        super().__init__(args, **kwargs)
        self.max_seq_length = 512
        self.model_config = 'data/pretrained_models/deberta-base-v2/model_config.json'
        self.vocab_type = 'BERT'
        self.vocab_path = 'data/pretrained_models/deberta-base-v2/vocab.txt'
        self.data_dir = data_dir
        self.args = args
        self.metric = 'bert'
        self.main_metric = 'perplexity'
        self.optimizer_name = 'Lamb'
        self.tokenizer = BertTokenizer(self.vocab_path)
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
        return create_dynamic_dataset(data, self.get_feature_fn(), 
                                      batch_size=batch_size,
                                      output_columns=output_columns, 
                                      repeat=self.args.num_train_epochs,
                                      num_workers=self.args.data_workers,
                                      num_shards=self.rank_size, 
                                      shard_id=self.rank_id)

    def eval_data(self, max_seq_len=512, batch_size=32, **kwargs):
        ...

    def get_metrics(self, **kwargs):
        """Calcuate metrics based on prediction results"""
        return OrderedDict(
            bert = BertMetric(self.args.eval_batch_size),
            )

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
            ......
            return tuple(features)

        return example_to_feature

    def load_data(self, path, type=None, max_seq_len=512):
        examples = []
        ......
        return ExampleSet(examples)


    def get_model(self):
        from MindsporeTrainer.modeling.models import BertPreTraining, DebertaPreTraining
        from MindsporeTrainer import build_transformer_model
        if self.args.fp16:
            compute_type = mstype.float16
        else:
            compute_type = mstype.float32
        model, config = build_transformer_model(self.model_config, 
                                                model=DebertaPreTraining, 
                                                compute_type=compute_type, 
                                                padding_idx=self.tokenizer._convert_token_to_id(self.tokenizer.pad_token))
        with open(os.path.join(self.args.output_dir, 'config.json'), 'w', encoding='utf-8') as f:
            f.write(config.to_json_string())
        copyfile(self.vocab_path, os.path.join(self.args.output_dir, 'vocab.txt'))
        return model
        # return partial_class

    def get_loss(self, *args, **kwargs):
        from MindsporeTrainer.modeling.loss import BertPretrainingLoss
        return BertPretrainingLoss(self.tokenizer.vocab_size)

    def get_eval_head(self, *args, **kwargs):
        from MindsporeTrainer.modeling.models import BertEvalHead
        return BertEvalHead(self.tokenizer.vocab_size)

    def get_opt_fn(self, *args, **kwargs):
        return None

```
### 2. 编写启动脚本
run.py
```
import MindsporeTrainer as mst
mst.launch()
```
### 3. 运行任务
```
python run.py --task_name=RESNETTask --do_train --do_eval --data_dir=data --num_train_epochs=10 --learning_rate=1e-2 --train_batch_size=64 --save_eval_steps=1000 --output_dir=output
```
## 多卡训练
官方推荐编写bash脚本，利用mpi进行训练，这里采用了纯python的实现。
### 定义必须的环境变量

bash
```
export RANK_SIZE = 8
export RANK_TABLE_FILE = /path/hccl.json
```
vscode调试环境
```
"env": {
    "RANK_SIZE": "8",
    "RANK_TABLE_FILE": "/path/hccl.json"
}
```
设置参数，开始训练
```
python run.py ...... --device_num=8 --device_id=0,1,2,3,4,5,6,7
```
# 模型创建
## 自定义模型
Mindspore相对TF和PyTorch，有其自身的特点，建模习惯也不同。在本框架中，为了便于模块化与代码复用，将模型分为**backbone**和**predict head**两部分。  
需要注意的是，模型定义中，construct函数返回都应当是tuple，即使只有一个对象返回，也应当采用(obj,)的形式。
### backbone
模型的主体部分定义。
### predict head
模型的附属部分，根据模型的用途，通常可以定义为loss层，或者在evaluation过程中定义的eval head，又或者其他用途的头。
## 使用官方[model_zoo](https://gitee.com/mindspore/models)中的模型
其代码结构如下
```shell
models
├── official                                    # 官方支持模型
│   └── XXX                                     # 模型名
│       ├── README.md                           # 模型说明文档
│       ├── requirements.txt                    # 依赖说明文件
│       ├── eval.py                             # 精度验证脚本
│       ├── export.py                           # 推理模型导出脚本
│       ├── scripts                             # 脚本文件
│       │   ├── run_distributed_train.sh        # 分布式训练脚本
│       │   ├── run_eval.sh                     # 验证脚本
│       │   └── run_standalone_train.sh         # 单机训练脚本
│       ├── src                                 # 模型定义源码目录
│       │   ├── XXXNet.py                       # 模型结构定义
│       │   ├── callback.py                     # 回调函数定义
│       │   ├── config.py                       # 模型配置参数文件
│       │   └── dataset.py                      # 数据集处理定义
│       ├── ascend_infer                        # （可选）用于在Ascend推理设备上进行离线推理的脚本
│       ├── third_party                         # （可选）第三方代码
│       │   └── XXXrepo                         # （可选）完整克隆自第三方仓库的代码
│       └── train.py                            # 训练脚本
├── research                                    # 非官方研究脚本
├── community                                   # 合作方脚本链接
└── utils                                       # 模型通用工具
```
找到所需的模型目录后，在src/xxxmodel.py中找到相应的定义，如果有定义好的backbone和head，那么可以直接引入使用。  
例如使用其中的BERT模型：  
`git clone https://gitee.com/mindspore/models`  
复制出bert源码到工作目录`models/official/nlp/bert/src`  
定义task
```
    from bert.src.bert_for_pre_training import BertPreTraining, BertPretrainingLoss
    ......
    def get_model(self):
        from MindsporeTrainer import build_transformer_model
        if self.args.fp16:
            compute_type = mstype.float16
        else:
            compute_type = mstype.float32
        model, config = build_transformer_model(self.model_config, 
                                                model=BertPreTraining, 
                                                compute_type=compute_type, 
                                                padding_idx=self.tokenizer._convert_token_to_id(self.tokenizer.pad_token))
        with open(os.path.join(self.output_dir, 'config.json'), 'w', encoding='utf-8') as f:
            f.write(config.to_json_string())
        copyfile(self.vocab_path, os.path.join(self.output_dir, 'vocab.txt'))
        return model

    def get_loss(self, *args, **kwargs):
        return BertPretrainingLoss(self.tokenizer.vocab_size)

    def get_eval_head(self, *args, **kwargs):
        from MindsporeTrainer.modeling.models import BertEvalHead
        return BertEvalHead(self.tokenizer.vocab_size)
```
# 参数介绍
训练超参数基本上都是通过运行参数来控制的，另外一些则可以在task定义中控制。
## commmon args
+ --accumulation_steps   
    type=int  
    default=1  
    Accumulating gradients N times before weight update, default is 1.
+ --allreduce_post_accumulation  
    type=str  
    default=true  
    choices=[true, false]  
    Whether to allreduce after accumulation of N steps or after each step, default is true.
+ --data_dir  
    default=None  
    type=str  
    required=False  
    The input data dir. Should contain the .tsv files (or other data files) for the task.
+ --data_sink_steps  
    type=int   
    default=1  
    Sink steps for each epoch, default is 1.

+ --do_train  
    default=False  
    action='store_true'  
    Whether to run training.

+ --do_eval  
    default=False  
    action='store_true'  
    Whether to run eval on the dev set.

+ --do_predict  
    default=False  
    action='store_true'  
    Whether to run prediction on the test set.

+ --debug  
    default=False  
    action='store_true'  
    Whether to cache cooked binary features

+ --device_target   
    type=str  
    default='Ascend'   
    choices=['Ascend   'GPU']  
    device where the code will be implemented. (Default: Ascend)

+ --distribute  
    default=False  
    action='store_true'  
    Run distribute, default is false.

+ --device_id  
  type=str  
  default=0  
  Device id, default is 0.
+ --device_num  
  type=int  
  default=1  
  Use device nums, default is 1.
+ --enable_data_sink  
    default=False  
    action='store_true'  
    Enable data sink, default is false.
+ --load_checkpoint_path  
    type=str  
    default=''  
    Load checkpoint file path
+ --num_train_epochs  
    default=1  
    type=int  
    Total number of training epochs to perform.
+ --output_dir  
    default=None  
    type=str  
    required=True  
    The output directory where the model checkpoints will be written.
+ --run_mode  
    type=str  
    default='GRAPH'  
    choices=['GRAPH   'PY']  
    0: GRAPH_MODE, 1: PY_NATIVE_MODE
+ --save_eval_steps  
    type=int  
    default=1000  
    Save checkpoint and evaluate steps, default is 1000.
+ --save_checkpoint_num  
    type=int  
    default=1  
    Save checkpoint numbers, default is 1.
+ --tag  
    type=str  
    default='final'  
    The tag name of current prediction/runs.
+ --task_dir  
    default=None  
    type=str  
    required=False  
    The directory to load customized tasks.
+ --task_name  
    default=None  
    type=str  
    action=LoadTaskAction  
    required=True  
    The name of the task to train.

## train args

+ --data_workers  
            type=int  
            default=4  
            The workers to load data.  
+ --enable_graph_kernel   
    type=str   
    default=auto   
    choices=[auto, true, false]  
    Accelerate by graph kernel, default is auto.
+ --eval_batch_size   
    default=32  
    type=int  
    Total batch size for eval.
+ --enable_global_norm  
            type=bool  
            default=False  
            enable global norm  
+ --predict_batch_size  
    default=32  
    type=int  
    Total batch size for prediction.
+ --report_interval  
            default=1  
            type=int  
            Interval steps for state report.  
+ --save_graphs   
    default=False  
    action='store_true'  
    Whether to save graphs
+ --seed  
            type=int  
            default=1234  
            random seed for initialization  
+ --thor   
    default=False  
    action='store_true' 
    Whether to convert model to thor optimizer
+ --train_batch_size  
            default=64  
            type=int  
            Total batch size for training.  
+ --train_steps  
    type=int  
    default=-1  
    Training Steps, default is -1, meaning run all steps according to epoch number.
## optimizer args
+ --fp16  
            default=False  
            type=boolean_string  
            Whether to use 16-bit float precision instead of 32-bit

+ --learning_rate  
            default=5e-5  
            type=float  
            The initial learning rate for Adam.

+ --loss_scale_value  
            type=int  
            default=1024  
            initial loss scale value  

+ --resume_opt_path  
            type=str.lower  
            default=''  
            The optimizer to be resumed.
+ --scale_factor  
            type=int  
            default=4  
            loss scale factor  

+ --scale_window  
            type=int  
            default=1000  
            loss window  

+ --warmup  
            default=0.1  
            type=float  
            Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10% of training.
## task中增加自定义参数
```
    @classmethod
    def add_arguments(cls, parser):
        """Add task specific arguments
            e.g. parser.add_argument('--data_dir', type=str, help='The path of data directory.')
        """
        parser.add_argument('--task_example_arg', type=str, default=None, help='An example task specific argument')

        return parser
```

# Task
所有的task都应当继承于`MindsporeTrainer.apps.tasks.Task`,   
为transformer定义的`MindsporeTrainer.apps.tasks.TransformerTask`也继承自Task。  
Task类的定义为:
```
class Task():
    _meta={}

    def __init__(self, args, **kwargs):
        self.args = args
    
    def eval_data(self, **kwargs):
        """
        Get eval dataset object.
        """
        return None

    def train_data(self, **kwargs):
        """
        Get train dataset object.
        """
        return None

    def test_data(self, **kwargs):
        return None

    def get_labels(self):
        """Gets the list of labels for this data set."""
        return None

    def get_eval_fn(self, *args, **kwargs):
        """
        Get the evaluation function
        """
        return None

    def get_eval_head(self, *args, **kwargs):
        """
        Get the evaluate head, the head replace loss function head when evaluation process
        """
        return None

    def get_pred_fn(self, *args, **kwargs):
        """
        Get the predict function
        """
        return None

    def get_loss(self, *args, **kwargs):
        """
        Get the loss function
        """
        return None

    def get_opt_fn(self, *args, **kwargs):
        """
        Get a function wich return the opimizer
        """
        def get_optimizer(*args, **kwargs):
            pass
        return get_optimizer

    def get_metrics(self):
        """Calcuate metrics based on prediction results"""
        return None

    def get_predict_fn(self):
        """Calcuate metrics based on prediction results"""
        def predict_fn(logits, output_dir, name, prefix):
            pass
        return None

    def get_feature_fn(self, **kwargs):
        """
        get the featurize function
        """
        def _example_to_feature(**kwargs):
             return feature
        return _example_to_feature
    
    def get_model(self):
        """
        Get a model instance
        """
        raise NotImplementedError('method not implemented yet.')

    @classmethod
    def add_arguments(cls, parser):
        """Add task specific arguments
        """
        pass
```
# API
## MindsporeTrainer
    MindsporeTrainer.launch()
    启动器，可支持分布式启动
    MindsporeTrainer.build_transformer_model(
                                            config_path=None,
                                            model='bert',
                                            application='encoder',
                                            **kwargs
                                            )
    创建transformer模型  
    args:   
        config_path model config 路径  
        model 为str的话，从预定义模型中获取模型类，为类名的话，直接进行实例化  
        application 用途，默认为'encoder'，TODO：实现decoder等  
        其他参数
## MindsporeTrainer.modeling
    建模模块，并提供若干预定义的模型，目前包括BERT和DeBERTa。
### MindsporeTrainer.modeling.models
    提供若干预定义的模型，目前包括BERT和DeBERTa。
### MindsporeTrainer.modeling.loss
    提供预定义的loss
### MindsporeTrainer.modeling.tokenizers
    提供预定义的tokenizers，目前仅支持BertTokenizer
## MindsporeTrainer.data 
    数据相关
### MindsporeTrainer.data.ExampleInstance
    样本实例
### MindsporeTrainer.data.ExampleSet
    样本集
### MindsporeTrainer.data.dynamic_dataset
    创建动态数据集
## MindsporeTrainer.utils
    各种实用组件
### MindsporeTrainer.utils.metrics
    提供多种自定义metric
### MindsporeTrainer.utils.masker
    用于生成MLM的mask
## MindsporeTrainer.apps.tasks
    任务相关
### MindsporeTrainer.apps.tasks.Task
    任务基类
### MindsporeTrainer.apps.tasks.TransformerTask
    Transformer任务类，继承自Task
## MindsporeTrainer.optims
    优化器、学习率调度等

# models
作者实现的模型，努力丰富中......
## DeBERTa
原论文：[DeBERTa: Decoding-enhanced BERT with Disentangled Attention](https://arxiv.org/abs/2006.03654)  
原仓库：[https://github.com/microsoft/DeBERTa](https://github.com/microsoft/DeBERTa)  
实现的是DeBERTa v2，详见[DeBERTa](./MindsporeTrainer/apps/tasks/deberta.py) task  
# 作者
周波 

DMAC Group@ZJU 浙江大学人工智能研究所