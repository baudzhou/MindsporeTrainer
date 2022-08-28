import copy
from loguru import logger
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from ...modeling import NNModule, ACT2FN, StableDropout, PoolConfig, ContextPooler, StableDropout
from MindsporeTrainer.modeling.gat import GatModel
from ...utils import *
from .xuci import WeightedLayerPooling

class GatForTokenClassification(NNModule):

    def __init__(self, config, num_labels, **kwargs):
        super(GatForTokenClassification, self).__init__(config)
        self.config = copy.deepcopy(config)
        self.num_labels = num_labels
        self.model = GatModel(config, pooler=False)

        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        drop_out = config.hidden_dropout_prob # if drop_out is None else drop_out
        self.dropout = StableDropout(drop_out)
        self.apply(self.init_weights)

    def forward(self, input_ids, input_mask=None, labels=None, position_ids=None, attention_mask=None, **kwags):
        device = list(self.parameters())[0].device
        input_ids = input_ids.to(device)
        if input_mask is None:
            input_mask = attention_mask
        input_mask = input_mask.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        sequence_output = self.model(input_ids, input_mask, token_type_ids=None, attention_mask=attention_mask,
                                                    output_all_encoded_layers=False)
        sequence_output = sequence_output['hidden_states'][-1]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output).float()
        loss = 0
        if labels is not None:
            valid_labels = labels.long().view(-1)
            label_index = (valid_labels>=0).nonzero().view(-1)
            valid_labels = valid_labels.index_select(dim=0, index=label_index)
            valid_logits = logits.view(-1, logits.size(-1)).index_select(dim=0, index=label_index)
            loss_fn = CrossEntropyLoss()
            loss = loss_fn(valid_logits, valid_labels)

        return {
                'logits' : logits,
                'loss' : loss,
                'labels': labels
            }


class GatForSequenceCls(NNModule):
    def __init__(self, config, num_labels, **kwargs):
        super(GatForSequenceCls, self).__init__(config)
        self.config = copy.deepcopy(config)
        self.num_labels = num_labels
        self.model = GatModel(config, pooler=False)
        pool_config = PoolConfig(self.config)
        self.pooler = ContextPooler(pool_config)
        self.classifier = nn.Linear(self.pooler.output_dim(), self.num_labels)
        drop_out = config.hidden_dropout_prob # if drop_out is None else drop_out
        self.dropout = StableDropout(drop_out)
        self.apply(self.init_weights)

    def forward(self, input_ids, input_mask=None, labels=None, position_ids=None, attention_mask=None, **kwags):
        device = list(self.parameters())[0].device
        input_ids = input_ids.to(device)
        if input_mask is None:
            input_mask = attention_mask
        input_mask = input_mask.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        sequence_output = self.model(input_ids, input_mask, token_type_ids=None, attention_mask=attention_mask,
                                                    output_all_encoded_layers=False)
        sequence_output = sequence_output['hidden_states'][-1]
        pooled_output = self.pooler(sequence_output)
        logits = self.classifier(self.dropout(pooled_output)).float()
        loss = 0
        if labels is not None:
            valid_labels = labels.long()
            loss_fn = CrossEntropyLoss()
            loss = loss_fn(logits, valid_labels)

        return {
                'logits' : logits,
                'loss' : loss,
                'labels': labels
            }



class GatForTlcModel(NNModule):
    def __init__(self, model_config, drop_out=None, **kwargs):
        super().__init__()
        self.config = model_config
        self.num_labels_0 = kwargs.get('num_labels_0', 1)
        self.num_labels_1 = kwargs.get('num_labels_1', 1)
        self.classifier_0 = nn.Linear(model_config.hidden_size, self.num_labels_0, bias=False)
        self.classifier_1 = nn.Linear(model_config.hidden_size, self.num_labels_1, bias=False)
        self.projection = nn.Linear(self.num_labels_0, self.num_labels_1)
        drop_out = model_config.hidden_dropout_prob if drop_out is None else drop_out
        self.dropout = StableDropout(drop_out)
        self.model = GatModel(model_config)
        self.loss_fn = CrossEntropyLoss()
        self.act = ACT2FN['gelu']
        pool_config = PoolConfig(self.config)
        self.pooler = ContextPooler(pool_config)
        self.apply(self.init_weights)


    def forward(self, input_ids, input_mask=None, labels=None, position_ids=None, attention_mask=None, **kwags):
        device = list(self.parameters())[0].device
        input_ids = input_ids.to(device)
        if input_mask is None:
            input_mask = attention_mask
        input_mask = input_mask.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        sequence_output = self.model(input_ids, input_mask, token_type_ids=None, attention_mask=attention_mask,
                                                    output_all_encoded_layers=False)
        sequence_output = sequence_output['hidden_states'][-1]
        pooled_output = self.pooler(sequence_output)
        logits_0 = self.classifier_0(self.dropout(pooled_output))
        labels = labels.long()
        loss_0 = self.loss_fn(logits_0, labels[:, 0])
        # prj = self.projection(logits_0)
        # prj = self.act(prj)
        logits_1 = self.classifier_1(self.dropout(pooled_output))
        # logits_1 = prj * logits_1
        loss_1 = self.loss_fn(logits_1, labels[:, 1])

        loss = loss_0 + loss_1

        return {
                'logits' : torch.cat([logits_0, logits_1], dim=1),
                'loss' : loss,
                'labels': labels
            }


class GatForMultiChoiceModel(NNModule):
    def __init__(self, model_config, num_labels = 2, drop_out=None, **kwargs):
            super().__init__(model_config)
            self.num_labels = num_labels
            self.config = model_config
            pool_config = PoolConfig(self.config)
            output_dim = model_config.hidden_size
            self.pooler = ContextPooler(pool_config)
            output_dim = self.pooler.output_dim()
            drop_out = model_config.hidden_dropout_prob if drop_out is None else drop_out
            self.cls = torch.nn.Linear(output_dim, 1)
            self.dropout = StableDropout(drop_out)
            self.model = GatModel(model_config)
            self.apply(self.init_weights)

    def forward(self, input_ids, input_mask=None, labels=None, position_ids=None, attention_mask=None, **kwags):
        num_opts = input_ids.size(1)
        input_ids = input_ids.view([-1, input_ids.size(-1)])
        if position_ids is not None:
            position_ids = position_ids.view([-1, position_ids.size(-1)])

        device = list(self.parameters())[0].device
        input_ids = input_ids.to(device)
        if input_mask is None:
            input_mask = attention_mask
        input_mask = input_mask.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.view([-1, attention_mask.size(-1)]).to(device)
        if input_mask is not None:
            input_mask = input_mask.view([-1, input_mask.size(-1)])

        sequence_output = self.model(input_ids, input_mask, token_type_ids=None, attention_mask=attention_mask,
                                                    output_all_encoded_layers=False)
        sequence_output = sequence_output['hidden_states'][-1]
        logits = self.cls(self.dropout(self.pooler(sequence_output)))
        logits = logits.float().squeeze(-1)
        logits = logits.view([-1, num_opts])
        loss = 0
        if labels is not None:
            labels = labels.long()
            loss_fn = CrossEntropyLoss()
            loss = loss_fn(logits, labels)

            return {
                    'logits' : logits,
                    'loss' : loss
                }

class GatForXuciModel(NNModule):
    def __init__(self, model_config, weight_layer_pooler=False, drop_out=None):
        super().__init__()
        self.config = model_config
        self.model = GatModel(model_config)
        if weight_layer_pooler:
            self.pooler = WeightedLayerPooling()
        else:
            self.pooler = None
        self.classifier = nn.Linear(model_config.hidden_size * 3, 2, bias=False)
        drop_out = model_config.hidden_dropout_prob if drop_out is None else drop_out
        self.dropout = StableDropout(drop_out)
        self.apply(self.init_weights)

    def forward(self, input_ids, compare_pos, position_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        # input_ids_a = input_ids[:, 0, ...].squeeze()
        # input_ids_b = input_ids[:, 1, ...].squeeze()
        # attention_mask_a = attention_mask[:, 0, ...].squeeze()
        # attention_mask_b = attention_mask[:, 1, ...].squeeze()
        device = list(self.parameters())[0].device
        input_ids = input_ids.view(-1, input_ids.size(-1))
        attention_mask = attention_mask.view(-1, attention_mask.size(-1))
        if input_mask is None:
            input_mask = attention_mask
        input_mask = input_mask.to(device)
        outputs = self.model(input_ids, input_mask, token_type_ids=None, attention_mask=attention_mask,
                                                    output_all_encoded_layers=False)
        prediction_scores = outputs['hidden_states'][-1]
        # sequence_output_b, _ = self.bert(input_ids_b, token_type_ids=token_type_ids, attention_mask=attention_mask_b)
        prediction_scores = prediction_scores.view(-1, 2, prediction_scores.size(1), prediction_scores.size(2))
        sequence_output_a, sequence_output_b = torch.unbind(prediction_scores, dim=1)
        # sequence_output_b = prediction_scores[:, 1, ...].squeeze()
        tokens_a = []
        tokens_b = []
        for so_a, so_b, pos in zip(sequence_output_a, sequence_output_b, compare_pos):
            a = so_a.index_select(0, pos[0][pos[0] > 0]).mean(0)
            b = so_b.index_select(0, pos[1][pos[1] > 0]).mean(0)
            tokens_a.append(a)
            tokens_b.append(b)
        
        tokens_a = torch.stack(tokens_a)
        tokens_b = torch.stack(tokens_b)
        
        # cosine similarity loss
        # cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        # logits = cos(tokens_a, tokens_b)
        # loss_fn = MSELoss()
        # loss = loss_fn(logits, labels.half())

        # according to SentenceBERT, use u,v,u-v as output
        scores = torch.cat([tokens_a, tokens_b, torch.abs(tokens_a - tokens_b)], -1)
        scores = self.dropout(scores)
        logits = self.classifier(scores).float()

        labels = labels.long().squeeze()

        loss_fn = CrossEntropyLoss() # 
        loss = loss_fn(logits, labels)

        return {
                'logits' : logits,
                'loss' : loss,
                'labels': labels
            }