from ast import Not
import types
from typing import Union, Tuple, List, Iterable, Dict
import torch
from torch import nn, Tensor
from torch.nn import CrossEntropyLoss

from MindsporeTrainer.modeling.ops import ACT2FN
from ...modeling import NNModule, StableDropout, PoolConfig, ContextPooler, StableDropout
from MindsporeTrainer.modeling.gat import GatModel


class TlcModel(NNModule):
    def __init__(self, model_config, mlm_model, weight_layer_pooler=False, drop_out=None, **kwargs):
        super().__init__()
        self.config = model_config
        self.num_labels_0 = kwargs.get('num_labels_0', 1)
        self.num_labels_1 = kwargs.get('num_labels_1', 1)
        pool_config = PoolConfig(self.config)
        self.pooler = ContextPooler(pool_config)
        output_dim = self.pooler.output_dim()
        self.classifier_0 = nn.Linear(output_dim, self.num_labels_0, bias=False)
        self.classifier_1 = nn.Linear(output_dim, self.num_labels_1, bias=False)
        self.projection = nn.Linear(self.num_labels_0, self.num_labels_1)
        drop_out = model_config.hidden_dropout_prob if drop_out is None else drop_out
        self.dropout = StableDropout(drop_out)
        self.apply(self.init_weights)
        self.bert = mlm_model.base_model
        self.loss_fn = CrossEntropyLoss()
        self.act = ACT2FN['gelu']

    def forward(self, input_ids, position_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        attention_mask = attention_mask.view(-1, attention_mask.size(-1))
        outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        prediction_scores = outputs['last_hidden_state']
        prediction_scores = self.pooler(prediction_scores)
        logits_0 = self.classifier_0(self.dropout(prediction_scores))
        labels = labels.long()
        loss_0 = self.loss_fn(logits_0, labels[:, 0])
        # prj = self.projection(logits_0)
        # prj = self.act(prj)
        logits_1 = self.classifier_1(self.dropout(prediction_scores))
        # logits_1 = prj * logits_1
        loss_1 = self.loss_fn(logits_1, labels[:, 1])

        loss = loss_0 + loss_1

        return {
                'logits' : torch.cat([logits_0, logits_1], dim=1),
                'loss' : loss,
                'labels': labels
            }
