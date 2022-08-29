# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# zbo@zju.edu.cn
# 2022-08-08
# ============================================================================


import math
import copy
from functools import lru_cache
from turtle import position
import numpy as np
import mindspore.common.dtype as mstype
import mindspore.nn as nn
import mindspore.ops.functional as F
from mindspore.common.initializer import TruncatedNormal, initializer
from mindspore.ops import operations as P
from mindspore.ops import composite as C
import mindspore.ops as O
from mindspore.common.tensor import Tensor
from mindspore.common.parameter import Parameter

from .layers import *


class Transformer(nn.Cell):
    """模型基类
    """
    def __init__(
        self,
        vocab_size,  # 词表大小
        hidden_size,  # 编码维度
        num_hidden_layers,  # Transformer总层数
        num_attention_heads,  # Attention的头数
        intermediate_size,  # FeedForward的隐层维度
        hidden_act,  # FeedForward隐层的激活函数
        max_position_embeddings = 512,
        dropout_rate=None,  # Dropout比例
        hidden_dropout_prob=0.1,
        initializer_range=0.02,
        attention_probs_dropout_prob=None,  # Attention矩阵的Dropout比例
        embedding_size=None,  # 是否指定embedding_size
        attention_head_size=None,  # Attention中V的head_size
        attention_key_size=None,  # Attention中Q,K的head_size
        sequence_length=None,  # 是否固定序列长度
        residual_attention_scores=False,  # Attention矩阵加残差
        type_vocab_size = 1,
        layer_norm_eps=1e-05,
        compute_type=mstype.float32,
        prefix=None,  # 层名前缀
        name=None,  # 模型名称
        config=None,
        **kwargs
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = attention_head_size or hidden_size // num_attention_heads
        self.attention_key_size = attention_key_size or self.attention_head_size
        self.intermediate_size = intermediate_size
        self.initializer_range = initializer_range
        self.hidden_dropout_prob = hidden_dropout_prob
        self.dropout_rate = dropout_rate or 0
        self.attention_probs_dropout_prob = attention_probs_dropout_prob or 0
        self.hidden_act = hidden_act
        self.embedding_size = embedding_size or hidden_size
        self.sequence_length = sequence_length
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.layer_norm_eps = layer_norm_eps
        self.attention_bias = None
        self.position_bias = None
        self.attention_scores = None
        self.residual_attention_scores = residual_attention_scores
        self.prefix = prefix or ''
        self.name = name
        self.compute_type = compute_type
        self.config = config
        self.embeddings = None
        self.encoder = None
        self.decoder = None
        self.output = None

    def build_layers(self, **kwargs):
        raise NotImplementedError

    def get_attention_bias(self, **kwargs):
        '''
        used to generate complex attention mask, UNILM etc.
        '''
        raise self.attention_bias

    def get_position_bias(self, **kwargs):
        raise self.position_bias

    def get_attention_mask(self, mask):
        # input mask from input index sequence
        if mask.ndim == 2: # (batch_size, seq_len)
            seq_length = F.shape(mask)[1]
            mask = F.cast(F.reshape(mask, (-1, 1, seq_length)), mstype.float32)
        return mask

    def parameter_name_mapping(self):
        return {}

    def construct(self, **kwargs):
        raise NotImplementedError


class BertModel(Transformer):
    """
    Bidirectional Encoder Representations from Transformers.

    Args:
        config (Class): Configuration for BertModel.
        is_training (bool): True for training mode. False for eval mode.
        use_one_hot_embeddings (bool): Specifies whether to use one hot encoding form. Default: False.
    """
    def __init__(self,
                 use_relative_positions=False,
                 gradient_checkpointing = False,
                 tokenizer = 'BertTokenizer',
                 use_pos=True,
                 config=None,
                 use_one_hot_embeddings=False,
                 **kwargs):
        super(BertModel, self).__init__(**kwargs)
        config = copy.deepcopy(config)

        self.last_idx = self.num_hidden_layers - 1
        # output_embedding_shape = [-1, self.seq_length, self.embedding_size]

        self.embeddings = Embeddings(use_relative_positions=use_relative_positions, **kwargs)

        self.encoder = Encoder(
            hidden_size=self.hidden_size,
            num_attention_heads=self.num_attention_heads,
            num_hidden_layers=self.num_hidden_layers,
            intermediate_size=self.intermediate_size,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            use_one_hot_embeddings=use_one_hot_embeddings,
            initializer_range=self.initializer_range,
            hidden_dropout_prob=self.hidden_dropout_prob,
            use_relative_positions=use_relative_positions,
            hidden_act=self.hidden_act,
            compute_type=self.compute_type,
            return_all_encoders=True,
            max_position_embeddings=self.max_position_embeddings,
            config=config)

        self.cast = P.Cast()
        self.dtype = self.compute_type
        self.cast_compute_type = SaturateCast(dst_type=self.compute_type)
        self.slice = P.StridedSlice()

        self.squeeze_1 = P.Squeeze(axis=1)
        self.pooler = nn.Dense(self.hidden_size, self.hidden_size,
                              activation="tanh",
                              weight_init=TruncatedNormal(self.initializer_range)).to_float(self.compute_type)
        self._create_attention_mask_from_input_mask = CreateAttentionMaskFromInputMask()

    def construct(self, input_ids, token_type_ids, input_mask):
        """Bidirectional Encoder Representations from Transformers."""
        # embedding
        embedding_tables = self.embeddings.word_embeddings.embedding_table
        word_embeddings = self.embeddings(input_ids, token_type_ids)['embeddings']

        # attention mask [batch_size, seq_length, seq_length]
        attention_mask = self._create_attention_mask_from_input_mask(input_mask)

        # bert encoder
        encoder_output = self.encoder(self.cast_compute_type(word_embeddings),
                                           attention_mask)['hidden_states']

        sequence_output = self.cast(encoder_output[self.last_idx], self.dtype)

        # pooler
        batch_size = P.Shape()(input_ids)[0]
        sequence_slice = self.slice(sequence_output,
                                    (0, 0, 0),
                                    (batch_size, 1, self.hidden_size),
                                    (1, 1, 1))
        first_token = self.squeeze_1(sequence_slice)
        pooled_output = self.pooler(first_token)
        pooled_output = self.cast(pooled_output, self.dtype)

        return sequence_output, pooled_output, embedding_tables


class BertPreTraining(nn.Cell):
    """
    Bert pretraining network.

    Args:
        config (BertConfig): The config of BertModel.
        is_training (bool): Specifies whether to use the training mode.
        use_one_hot_embeddings (bool): Specifies whether to use one-hot for embeddings.

    Returns:
        Tensor, prediction_scores, seq_relationship_score.
    """

    def __init__(self, **config):
        super(BertPreTraining, self).__init__()
        self.bert = BertModel(**config)
        self.cls1 = GetMaskedLMOutput(**config)
        self.cls2 = GetNextSentenceOutput(**config)

    def construct(self, *sample):
        '''
            sample: ["input_ids", "input_mask", "token_type_id", "next_sentence_labels",
                      "masked_lm_positions", "masked_lm_ids", "masked_lm_weights"]
        '''
        input_ids, input_mask, token_type_id = sample[: 3]
        masked_lm_positions = sample[4]
        sequence_output, pooled_output, embedding_table = \
            self.bert(input_ids, token_type_id, input_mask)
        prediction_scores = self.cls1(sequence_output,
                                      embedding_table,
                                      masked_lm_positions)
        seq_relationship_score = self.cls2(pooled_output)
        return prediction_scores, seq_relationship_score


class Deberta(Transformer):
    """
    """
    def __init__(self,
                 config=None,
                 use_one_hot_embeddings=False,
                 **kwargs):
        super(Deberta, self).__init__(**kwargs)
        config = copy.deepcopy(config)

        self.last_idx = self.num_hidden_layers - 1
        # output_embedding_shape = [-1, self.seq_length, self.embedding_size]

        self.embeddings = Embeddings(**kwargs)

        self.encoder = Encoder(
            hidden_size=self.hidden_size,
            num_attention_heads=self.num_attention_heads,
            num_hidden_layers=self.num_hidden_layers,
            intermediate_size=self.intermediate_size,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            use_one_hot_embeddings=use_one_hot_embeddings,
            initializer_range=self.initializer_range,
            hidden_dropout_prob=self.hidden_dropout_prob,
            use_relative_positions=kwargs.get('relative_attention', False),
            hidden_act=self.hidden_act,
            attention=DisentangledSelfAttention(self.__dict__, config=config),
            compute_type=self.compute_type,
            return_all_encoders=True,
            max_position_embeddings=self.max_position_embeddings,
            config=config)
        self.cast = P.Cast()
        self.dtype = self.compute_type
        self.cast_compute_type = SaturateCast(dst_type=self.compute_type)
        self.slice = P.StridedSlice()

        self.squeeze_1 = P.Squeeze(axis=1)
        self.pooler = nn.Dense(self.hidden_size, self.hidden_size,
                              activation="tanh",
                              weight_init=TruncatedNormal(self.initializer_range)).to_float(self.compute_type)
        self._create_attention_mask_from_input_mask = CreateAttentionMaskFromInputMask()

    def construct(self, input_ids, token_type_ids, input_mask, output_all_encoded_layers=True):
        """Bidirectional Encoder Representations from Transformers."""
        # embedding
        embedding_tables = self.embeddings.word_embeddings.embedding_table

        embeddings = self.embeddings(input_ids, token_type_ids)
        word_embeddings = embeddings['embeddings']
        position_embeddings = embeddings['position_embeddings']

        # attention mask [batch_size, seq_length, seq_length]
        attention_mask = self._create_attention_mask_from_input_mask(input_mask)

        # bert encoder
        encoder_output = self.encoder(self.cast_compute_type(word_embeddings),
                                      attention_mask, output_all_encoded_layers=output_all_encoded_layers)['hidden_states']
        # if output_all_encoded_layers:
        #     sequence_output = [self.cast(eo, self.dtype) for eo in encoder_output]
        # else:
        #     sequence_output = [self.cast(encoder_output[self.last_idx], self.dtype)]

        # pooler
        # encoder_output = tuple(encoder_output)
        batch_size = P.Shape()(input_ids)[0]
        sequence_slice = self.slice(encoder_output[-1],
                                    (0, 0, 0),
                                    (batch_size, 1, self.hidden_size),
                                    (1, 1, 1))
        first_token = self.squeeze_1(sequence_slice)
        pooled_output = self.pooler(first_token)
        pooled_output = self.cast(pooled_output, self.dtype)

        return encoder_output, pooled_output, embedding_tables, position_embeddings


class DebertaPreTraining(nn.Cell):
    """
    DeBerta pretraining network.

    Args:
        config (BertConfig): The config of BertModel.
        is_training (bool): Specifies whether to use the training mode.
        use_one_hot_embeddings (bool): Specifies whether to use one-hot for embeddings.

    Returns:
        Tensor, prediction_scores, seq_relationship_score.
    """

    def __init__(self, **config):
        super(DebertaPreTraining, self).__init__()
        self.config = config.get('config')
        self.bert = Deberta(**config)
        self.cls1 = GetMaskedLMOutput(**config)
        self.lm_predictions = EnhancedMaskDecoder(**config)
        self.pooled_loss = Tensor(np.zeros((1, 2)))

    def construct(self, *sample):
        '''
            sample: ["input_ids", "input_mask", "token_type_id", "next_sentence_labels",
                      "masked_lm_positions", "masked_lm_ids", "masked_lm_weights"]
        '''
        input_ids, input_mask, token_type_id = sample[: 3]
        masked_lm_positions, masked_lm_ids, masked_lm_weights = sample[4: 7]
        sequence_output, pooled_output, embedding_table, position_embeddings = \
            self.bert(input_ids, token_type_id, input_mask)
        z_states = position_embeddings
        pred_score = self.lm_predictions(sequence_output, z_states, input_mask, self.bert.encoder.layer[-1],
                                         self.bert.encoder.rel_embeddings.embedding_table,
                                         self.bert.encoder.relative_attention, self.bert.encoder.norm_rel_ebd, self.bert.encoder.LayerNorm)
        prediction_scores = self.cls1(pred_score,
                                      embedding_table,
                                      masked_lm_positions)
        return prediction_scores, self.pooled_loss.repeat(pred_score.shape[0], 0)