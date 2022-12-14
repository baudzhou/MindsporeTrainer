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
from shutil import copyfile
from loguru import logger

from MindsporeTrainer.data import ExampleInstance, ExampleSet, _truncate_segments
from MindsporeTrainer.data.example import *
from MindsporeTrainer.task import TransformerTask
from MindsporeTrainer.task  import register_task
from MindsporeTrainer.utils.metrics import *
from MindsporeTrainer.utils.metrics import BertMetric
from MindsporeTrainer.utils.masker import NGramMaskGenerator
from MindsporeTrainer.data.dynamic_dataset import create_dynamic_dataset
from MindsporeTrainer.modeling.tokenizers import BertTokenizer

@register_task(name="DEBERTA", desc="Basic DEBERTA task")
class DEBERTATask(TransformerTask):
    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)
        self.max_seq_len = 512
        self.model_config = 'config.json'
        self.vocab_type = 'BERT'
        self.vocab_path = 'vocab.txt'
        self.metric = 'bert'
        self.main_metric = 'perplexity'
        self.optimizer_name = 'Lamb'
        self.tokenizer = BertTokenizer(self.vocab_path)

    def train_data(self, **kwargs):
        # data_path = os.path.join(self.data_dir, 'daizhige.pkl')
        # data = self.load_data(data_path)
        data_path = os.path.join(self.data_dir, 'eval.txt')
        data = self.load_txt_data(data_path)
        # data = ExampleSet(data)
        output_columns = ["input_ids", "input_mask", "token_type_id", "next_sentence_labels",
                                    "masked_lm_positions", "masked_lm_ids", "masked_lm_weights"]
        return create_dynamic_dataset(data, [[self.get_feature_fn(), ["example"], output_columns]], 
                                      batch_size=self.train_batch_size,
                                      output_columns=output_columns, 
                                      repeat=self.num_train_epochs,
                                      num_workers=self.data_workers,
                                      num_shards=self.rank_size, 
                                      shard_id=self.rank_id,
                                      type_cast_op=None)

    def eval_data(self, **kwargs):
        data_path = os.path.join(self.data_dir, 'eval.txt')
        data = self.load_txt_data(data_path)

        output_columns = ["input_ids", "input_mask", "token_type_id", "next_sentence_labels",
                                    "masked_lm_positions", "masked_lm_ids", "masked_lm_weights"]
        return create_dynamic_dataset(data, [[self.get_feature_fn(), ["example"], output_columns]], 
                                      batch_size=self.eval_batch_size,
                                      output_columns=output_columns, 
                                      repeat=1,
                                      num_workers=self.data_workers, 
                                      num_shards=self.rank_size, 
                                      shard_id=self.rank_id,
                                      type_cast_op=None)

    def get_metrics(self, **kwargs):
        """Calcuate metrics based on prediction results"""
        return OrderedDict(
            bert = BertMetric(self.eval_batch_size),
            )

    def get_eval_fn(self, **kwargs):
        # data = kwargs.get('data')

        def run_eval(model, data, name, prefix):
            '''
            args: 
                model: Model instance
                name: evaluate name
                prefix: prefix of file
            return:
                float, main metric of this task, used to save best metric model
            '''
            if data is None:
                data = self.eval_data(**kwargs)
            res = model.eval(data, dataset_sink_mode=False)
            res = res['bert']
            main_metric = res[self.main_metric]
            if self.rank_id == 0:
                output=os.path.join(self.output_dir, 'submit-{}-{}.tsv'.format(name, prefix))
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


    def get_feature_fn(self, ext_params=None, rng=None, **kwargs):
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
            max_num_tokens = self.max_seq_len - 2
            segments = example[0].tolist().decode('utf-8')
            _tokens = ['[CLS]'] + tokenizer.tokenize(segments) + ['[SEP]']
            if mask_generator:
                    tokens, lm_labels = mask_generator.mask_tokens(_tokens, r)
            token_ids = tokenizer.convert_tokens_to_ids(tokens)
            pad_id = tokenizer._convert_token_to_id(tokenizer.pad_token)
            seq_len = len(token_ids)
            pad_num = self.max_seq_len - seq_len
            token_ids = token_ids + [pad_id] * pad_num
            input_mask = [1] * seq_len + [0] * pad_num
            token_type_id = [0] * self.max_seq_len
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

            features = [np.array(f, dtype=np.int32) for f in features]
            return tuple(features)

        return example_to_feature

    def load_txt_data(self, path, type=None):
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
                        if len(current_chunk) >= self.max_seq_len:
                            example = ExampleInstance(segments=[current_chunk[: self.max_seq_len - 2]])
                            if len(example.segments[0]) < 510:
                                print(len(example.segments[0]))
                            examples.append(example)
                            current_chunk = current_chunk[self.max_seq_len - 2: ]
        return ExampleSet(examples)

    def load_data(self, path, type=None):
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
                    if len(l) == 0:
                        continue
                    current_chunk += l
                    if len(current_chunk) >= self.max_seq_len:
                        if current_chunk[0] in ['O', '???', '???', '???', '???', '???', '???', '???', '|']:
                            current_chunk = current_chunk[1:]
                        example = ExampleInstance(segments=[current_chunk[: self.max_seq_len - 2]])
                        if len(example.segments[0]) < 510:
                            print(len(example.segments[0]))
                        # examples.append(np.array([current_chunk[: max_seq_len - 2]]))
                        examples.append(example)
                        current_chunk = current_chunk[self.max_seq_len - 2: ]
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
        with open(os.path.join(self.output_dir, 'config.json'), 'w', encoding='utf-8') as f:
            f.write(config.to_json_string())
        copyfile(self.vocab_path, os.path.join(self.output_dir, 'vocab.txt'))
        return model
        # return partial_class

    def get_loss(self, *args, **kwargs):
        from MindsporeTrainer.modeling.loss import BertPretrainingLoss
        return BertPretrainingLoss(self.tokenizer.vocab_size)

    def get_eval_head(self, *args, **kwargs):
        from MindsporeTrainer.modeling.layers import BertEvalHead
        return BertEvalHead(self.tokenizer.vocab_size)

    def get_opt_fn(self, *args, **kwargs):
        return None


