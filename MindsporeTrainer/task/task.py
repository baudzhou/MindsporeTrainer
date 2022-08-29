# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# zbo@zju.edu.cn
# 2022-08-08
# ============================================================================

import os
import csv
from collections import OrderedDict
import numpy as np
from utils.metrics import *
from mindspore.communication import get_rank, get_group_size

from MindsporeTrainer.modeling.layers import FakeHead


__all__ = ['EvalData', 'Task', 'TransformerTask']

class EvalData:
    def __init__(self, name, examples, metrics_fn=None, predict_fn=None, ignore_metric=False, critial_metrics=None):
        def accuracy_fn(logits, labels):
            return OrderedDict(accuracy= metric_accuracy(logits, labels))

        def default_pred_fn(logits, output_dir, name, prefix):
            output=os.path.join(output_dir, 'submit-{}-{}.tsv'.format(name, prefix))
            preds = np.argmax(logits, axis=-1)
            with open(output, 'w', encoding='utf-8') as fs:
                fs.write('index\tpredictions\n')
                for i,p in enumerate(preds):
                    fs.write('{}\t{}\n'.format(i, p))
        self.name = name
        self.data = examples
        self.ignore_metric = ignore_metric
        self.critial_metrics = critial_metrics
        self.metrics_fn = metrics_fn if metrics_fn is not None else accuracy_fn
        self.predict_fn = predict_fn if predict_fn is not None else default_pred_fn

    def __repr__(self):
        return f'{self.name}, {type(self.data)}: {len(self.data)}, {self.predict_fn}, {self.metrics_fn}'

class Task():
    _meta={}

    def __init__(self, args, **kwargs):
        self.args = args
        self.data_dir = args.data_dir
        self.num_train_epochs = self.args.num_train_epochs
        self.data_workers = self.args.data_workers
        self.train_batch_size = self.args.train_batch_size
        self.eval_batch_size = self.args.eval_batch_size
        self.predict_batch_size = self.args.predict_batch_size
        self.output_dir = self.args.output_dir
        if args.distribute:
            self.rank_id = get_rank()
            self.rank_size = get_group_size()
        else:
            self.rank_id = 0
            self.rank_size = 1

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

    def label2id(self, labelstr):
        label_dict = {l:i for i,l in enumerate(self.get_labels())}
        return label_dict[labelstr] if labelstr in label_dict else -1

    def get_train_fn(self, *args, **kwargs):
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
        return FakeHead()

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
            output=os.path.join(output_dir, 'submit-{}-{}.tsv'.format(name, prefix))
            preds = np.argmax(logits, axis=-1)
            preds = preds.flatten()
            labels = self.get_labels()
            with open(output, 'w', encoding='utf-8') as fs:
                fs.write('index\tpredictions\n')
                for i,p in enumerate(preds):
                    fs.write('{}\t{}\n'.format(i, labels[p]))

        return predict_fn

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding='utf-8') as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

    def get_feature_fn(self, max_seq_len = 512, mask_gen = None, label_type='int', training=False):
        # tokenizer = self.tokenizer
        # def _example_to_feature(example, rng=None, ext_params=None, **kwargs):
        #     return example_to_feature(tokenizer, example, max_seq_len = max_seq_len, \
        #         rng = rng, mask_generator = mask_gen, ext_params = ext_params, label_type=label_type, **kwargs)
        # return _example_to_feature
        return None

    def get_model_class_fn(self):
        # def partial_class(*wargs, **kwargs):
        #     model, tokenizer, model_config = NNModule.load_model(*wargs, self.model_class, **kwargs)
        #     self.tokenizer = tokenizer
        #     self.model_config = model_config
        #     if 'ForMaskedLM' in model.__class__.__name__:
        #         model = self.custom_model_class(model_config, model, num_labels=kwargs['num_labels'])
        #     return model
        # return partial_class
        return None
    
    def get_model(self):
        """
        Get a model instance
        """
        raise NotImplementedError('method not implemented yet.')

    @classmethod
    def add_arguments(cls, parser):
        """Add task specific arguments
            e.g. parser.add_argument('--data_dir', type=str, help='The path of data directory.')
        """
        # parser.add_argument('--task_example_arg', type=str, default=None, help='An example task specific argument')

        return parser


class TransformerTask(Task):
    _meta={}
    max_seq_len = 512
    model_config = ''
    vocab_type = ''
    vocab_path = ''
    def __init__(self, args, **kwargs):
        self.args = args
        super().__init__(args, **kwargs)