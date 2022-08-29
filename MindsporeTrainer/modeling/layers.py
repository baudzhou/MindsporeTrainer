# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# zbo@zju.edu.cn
# 2022-08-08
# ============================================================================

from distutils.command.config import config
import math
import copy
from collections import Sequence
import mindspore

import numpy
import mindspore.numpy as np
import mindspore.common.dtype as mstype
import mindspore.nn as nn
import mindspore.ops.functional as F
from mindspore.common.initializer import TruncatedNormal, initializer
from mindspore.ops import operations as P
from mindspore.ops import composite as C
import mindspore.ops as O
from mindspore.common.tensor import Tensor
from mindspore.common.parameter import Parameter
from mindspore import ms_function

from MindsporeTrainer.modeling.config import ModelConfig


class CreateAttentionMaskFromInputMask(nn.Cell):
    """
    Create attention mask according to input mask.

    Args:
        config (Class): Configuration for BertModel.
    """
    def __init__(self):
        super(CreateAttentionMaskFromInputMask, self).__init__()
        self.input_mask = None

        self.cast = P.Cast()
        self.reshape = P.Reshape()

    def construct(self, input_mask):
        seq_length = F.shape(input_mask)[1]
        attention_mask = self.cast(self.reshape(input_mask, (-1, 1, seq_length)), mstype.float32)
        return attention_mask


class Embeddings(nn.Cell):
    def __init__(self,
                 use_one_hot_embeddings=False,
                 **kwargs):
        super().__init__()
        vocab_size = kwargs.get('vocab_size')
        embedding_size = kwargs.get('hidden_size')
        hidden_size = kwargs.get('hidden_size')
        type_vocab_size = kwargs.get('type_vocab_size')
        hidden_dropout_prob = kwargs.get('hidden_dropout_prob')
        padding_idx = kwargs.get('padding_idx')
        max_position_embeddings = kwargs.get('max_position_embeddings')
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.use_relative_positions = kwargs.get('use_relative_positions')
        self.word_embeddings = nn.Embedding(vocab_size, embedding_size, padding_idx=padding_idx, use_one_hot=use_one_hot_embeddings,
                                            embedding_table='truncatedNormal')
        self.position_biased_input = kwargs.get('position_biased_input')
        if type_vocab_size > 0:
            self.use_token_type = True
            self.token_type_embeddings = nn.Embedding(type_vocab_size, embedding_size)
        else:
            self.use_token_type = False
        self.type_vocab_size = type_vocab_size
        if embedding_size != hidden_size:
            self.embed_proj = nn.Dense(embedding_size, hidden_size, has_bias=False)
        self.position_embeddings = nn.Embedding(
            vocab_size=max_position_embeddings,
            embedding_size=embedding_size,
            use_one_hot=False)
        ln_eps = kwargs.get('layer_norm_eps', 1e-7)
        self.LayerNorm = nn.LayerNorm((hidden_size, ), epsilon=ln_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.add = P.Add()

    def construct(self, input_ids, token_type_ids, mask=None, **kwargs):
        seq_length = input_ids.shape[1]

        position_ids = np.arange(0, seq_length).astype(mstype.int32)
        # position_ids = Tensor(np.arange(0, seq_length), dtype=mstype.int32)
        position_ids = O.BroadcastTo(input_ids.shape)(F.expand_dims(position_ids, 0))

        # if token_type_ids is None:
        #     token_type_ids = O.ZerosLike()(input_ids)

        output = self.word_embeddings(input_ids)
        
        if self.use_token_type:
            token_type_embeddings = self.token_type_embeddings(token_type_ids)
            output = self.add(output, token_type_embeddings)
        # else:
        #     output = self.add(output, 0)
        
        # if not self.use_relative_positions:
        position_embeddings = self.position_embeddings(position_ids)
        output = self.add(output, position_embeddings)
        # else:
        #     position_embeddings = None

        if self.embedding_size != self.hidden_size:
            output = self.embed_proj(output)
        # else:
        #     output = self.add(output, 0)
        embeddings = self.LayerNorm(output)
        embeddings = self.dropout(embeddings)
        return {
            'embeddings': embeddings,
            'position_embeddings': position_embeddings}   

class EmbeddingLookup(nn.Cell):
    """
    A embeddings lookup table with a fixed dictionary and size.

    Args:
        vocab_size (int): Size of the dictionary of embeddings.
        embedding_size (int): The size of each embedding vector.
        embedding_shape (list): [batch_size, seq_length, embedding_size], the shape of
                         each embedding vector.
        use_one_hot_embeddings (bool): Specifies whether to use one hot encoding form. Default: False.
        initializer_range (float): Initialization value of TruncatedNormal. Default: 0.02.
    """
    def __init__(self,
                 vocab_size,
                 embedding_size,
                 embedding_shape,
                 use_one_hot_embeddings=False,
                 initializer_range=0.02):
        super(EmbeddingLookup, self).__init__()
        self.vocab_size = vocab_size
        self.use_one_hot_embeddings = use_one_hot_embeddings
        self.embedding_table = Parameter(initializer
                                         (TruncatedNormal(initializer_range),
                                          [vocab_size, embedding_size]))
        self.expand = P.ExpandDims()
        self.shape_flat = (-1,)
        self.gather = P.Gather()
        self.one_hot = P.OneHot()
        self.on_value = Tensor(1.0, mstype.float32)
        self.off_value = Tensor(0.0, mstype.float32)
        self.array_mul = P.MatMul()
        self.reshape = P.Reshape()
        self.shape = tuple(embedding_shape)

    def construct(self, input_ids):
        """Get output and embeddings lookup table"""
        extended_ids = self.expand(input_ids, -1)
        flat_ids = self.reshape(extended_ids, self.shape_flat)
        if self.use_one_hot_embeddings:
            one_hot_ids = self.one_hot(flat_ids, self.vocab_size, self.on_value, self.off_value)
            output_for_reshape = self.array_mul(
                one_hot_ids, self.embedding_table)
        else:
            output_for_reshape = self.gather(self.embedding_table, flat_ids, 0)
        output = self.reshape(output_for_reshape, self.shape)
        return output, self.embedding_table


class EmbeddingPostprocessor(nn.Cell):
    """
    Postprocessors apply positional and token type embeddings to word embeddings.

    Args:
        embedding_size (int): The size of each embedding vector.
        embedding_shape (list): [batch_size, seq_length, embedding_size], the shape of
                         each embedding vector.
        use_token_type (bool): Specifies whether to use token type embeddings. Default: False.
        token_type_vocab_size (int): Size of token type vocab. Default: 16.
        use_one_hot_embeddings (bool): Specifies whether to use one hot encoding form. Default: False.
        initializer_range (float): Initialization value of TruncatedNormal. Default: 0.02.
        max_position_embeddings (int): Maximum length of sequences used in this
                                 model. Default: 512.
        dropout_prob (float): The dropout probability. Default: 0.1.
    """
    def __init__(self,
                 embedding_size,
                 embedding_shape,
                 use_relative_positions=False,
                 use_token_type=False,
                 token_type_vocab_size=16,
                 use_one_hot_embeddings=False,
                 initializer_range=0.02,
                 max_position_embeddings=512,
                 dropout_prob=0.1):
        super(EmbeddingPostprocessor, self).__init__()
        self.use_token_type = use_token_type
        self.token_type_vocab_size = token_type_vocab_size
        self.use_one_hot_embeddings = use_one_hot_embeddings
        self.max_position_embeddings = max_position_embeddings
        self.token_type_embedding = nn.Embedding(
            vocab_size=token_type_vocab_size,
            embedding_size=embedding_size,
            use_one_hot=use_one_hot_embeddings)
        self.shape_flat = (-1,)
        self.one_hot = P.OneHot()
        self.on_value = Tensor(1.0, mstype.float32)
        self.off_value = Tensor(0.1, mstype.float32)
        self.array_mul = P.MatMul()
        self.reshape = P.Reshape()
        self.shape = tuple(embedding_shape)
        self.dropout = nn.Dropout(1 - dropout_prob)
        self.gather = P.Gather()
        self.use_relative_positions = use_relative_positions
        self.slice = P.StridedSlice()
        _, seq, _ = self.shape
        self.full_position_embedding = nn.Embedding(
            vocab_size=max_position_embeddings,
            embedding_size=embedding_size,
            use_one_hot=False)
        self.layernorm = nn.LayerNorm((embedding_size,))
        self.position_ids = Tensor(np.arange(seq).reshape(-1, seq).astype(np.int32))
        self.add = P.Add()

    def construct(self, token_type_ids, word_embeddings):
        """Postprocessors apply positional and token type embeddings to word embeddings."""
        output = word_embeddings
        if self.use_token_type:
            token_type_embeddings = self.token_type_embedding(token_type_ids)
            output = self.add(output, token_type_embeddings)
        if not self.use_relative_positions:
            shape = F.shape(output)
            position_ids = self.position_ids[:, :shape[1]]
            position_embeddings = self.full_position_embedding(position_ids)
            output = self.add(output, position_embeddings)
        output = self.layernorm(output)
        output = self.dropout(output)
        return output


class DisentangledSelfAttention(nn.Cell):
    def __init__(self, config_dict=None, compute_type=mstype.float16, config=None, has_attention_mask=False):
        super().__init__()
        if config:
            self.config = config
        else:
            self.config = ModelConfig.from_dict(config_dict)
        self.has_attention_mask = has_attention_mask
        self.num_attention_heads = self.config.num_attention_heads
        _attention_head_size = int(self.config.hidden_size / self.config.num_attention_heads)
        self.attention_head_size = getattr(self.config, 'attention_head_size', _attention_head_size)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        initializer_range = self.config.initializer_range
        act_fn = self.config.hidden_act
        weight = TruncatedNormal(initializer_range)

        self.query_proj = nn.Dense(self.config.hidden_size, 
                                    self.all_head_size, 
                                    activation=act_fn, 
                                    weight_init=weight).to_float(compute_type)
        self.key_proj = nn.Dense(self.config.hidden_size, 
                                    self.all_head_size, 
                                    activation=act_fn, 
                                    weight_init=weight).to_float(compute_type)
        self.value_proj = nn.Dense(self.config.hidden_size, 
                                    self.all_head_size, 
                                    activation=act_fn, 
                                    weight_init=weight).to_float(compute_type)

        self.share_att_key = getattr(self.config, 'share_att_key', False)
        self.pos_att_type = [x.strip() for x in getattr(self.config, 'pos_att_type', 'c2p').lower().split('|')] # c2p|p2c
        self.relative_attention = getattr(self.config, 'relative_attention', False)

        if self.relative_attention:
            self.position_buckets = getattr(self.config, 'position_buckets', -1)
            self.max_relative_positions = getattr(self.config, 'max_relative_positions', -1)
            if self.max_relative_positions < 1:
                self.max_relative_positions = self.config.max_position_embeddings
            self.pos_ebd_size = self.max_relative_positions
            if self.position_buckets > 0:
                self.pos_ebd_size = self.position_buckets
                # For backward compitable

            self.pos_dropout = nn.Dropout(self.config.hidden_dropout_prob)

            if (not self.share_att_key):
                if 'c2p' in self.pos_att_type or 'p2p' in self.pos_att_type:
                    self.pos_key_proj = nn.Dense(self.config.hidden_size, 
                                                self.all_head_size, 
                                                activation=act_fn, 
                                                weight_init=weight).to_float(compute_type)
                if 'p2c' in self.pos_att_type or 'p2p' in self.pos_att_type:
                    self.pos_query_proj = nn.Dense(self.config.hidden_size, 
                                                    self.all_head_size, 
                                                    activation=act_fn, 
                                                    weight_init=weight,
                                                    has_bias=False).to_float(compute_type)

        self.dropout = nn.Dropout(self.config.attention_probs_dropout_prob)

        self.shape_return = (-1, self.num_attention_heads * self.attention_head_size )
        self.cast_compute_type = SaturateCast(dst_type=compute_type)
        self.trans_shape = (0, 2, 1, 3)
        self.trans_shape_relative = (2, 0, 1, 3)
        self.trans_shape_position = (1, 2, 0, 3)
        self.multiply_data = -10000.0
        self.scores_mul = 1.0 / math.sqrt(float(self.attention_head_size ))
        self.reshape = P.Reshape()
        self.shape_from_2d = (-1, self.config.hidden_size)
        self.shape_to_2d = (-1, self.config.hidden_size)
        self.matmul_trans_b = P.BatchMatMul(transpose_b=True)
        self.multiply = P.Mul()
        self.transpose = P.Transpose()
        self.matmul = P.BatchMatMul()
        self.softmax = nn.Softmax()
        self.dropout = nn.Dropout(self.config.attention_probs_dropout_prob)
        self.gather = O.GatherD()
        self.depend = P.Depend()

        # if self.has_attention_mask:
        self.expand_dims = P.ExpandDims()
        self.sub = P.Sub()
        self.add = P.Add()
        self.cast = P.Cast()
        self.get_dtype = P.DType()

    def transpose_for_scores(self, x, dense, attention_heads, target_shape, seq_len):
        x = self.reshape(x, target_shape)
        x = dense(x)
        x = self.reshape(x, (-1, seq_len, attention_heads, self.attention_head_size))
        x = self.transpose(x, self.trans_shape).view(-1, x.shape[1], x.shape[-1])
        return x

    def construct(self, hidden_states, query_states, attention_mask, return_att=False, relative_pos=None, rel_embeddings=None, **kwargs):
        if query_states is None:
            query_states = hidden_states
        
        shape_from = F.shape(attention_mask)[2]
        query_states = self.depend(query_states, shape_from)
        query_layer = self.transpose_for_scores(query_states, self.query_proj, self.num_attention_heads, self.shape_from_2d, shape_from)
        key_layer = self.transpose_for_scores(hidden_states, self.key_proj, self.num_attention_heads, self.shape_to_2d, shape_from)
        value_layer = self.transpose_for_scores(hidden_states, self.value_proj, self.num_attention_heads, self.shape_to_2d, shape_from)
       
        rel_att = None
        # Take the dot product between "query" and "key" to get the raw attention scores.
        scale_factor = 1
        if 'c2p' in self.pos_att_type:
            scale_factor += 1
        if 'p2c' in self.pos_att_type:
            scale_factor += 1
        if 'p2p' in self.pos_att_type:
            scale_factor += 1
        # scale = 1 / numpy.sqrt(query_layer.shape[-1] * scale_factor)
        attention_scores = self.matmul(query_layer, key_layer.transpose(0, 2, 1) * self.scores_mul)
        # if self.relative_attention:
        #     rel_embeddings = self.pos_dropout(rel_embeddings)
        #     rel_att = self.disentangled_attention_bias(query_layer, key_layer, relative_pos, rel_embeddings, scale_factor)

        if rel_att is not None:
            attention_scores = (attention_scores + rel_att)
        attention_scores = (attention_scores - attention_scores.max(-1, keepdims=True)).astype(hidden_states.dtype)
        attention_scores = attention_scores.view(-1, self.num_attention_heads, attention_scores.shape[-2], attention_scores.shape[-1])

        # bxhxlxd
        # if self.has_attention_mask:
        attention_mask = self.expand_dims(attention_mask, 1)
        multiply_out = self.sub(self.cast(F.tuple_to_array((1.0,)), self.get_dtype(attention_scores)),
                                self.cast(attention_mask, self.get_dtype(attention_scores)))
        # adder = self.multiply(multiply_out, self.multiply_data)
        attention_scores = self.add(self.multiply(multiply_out, self.multiply_data), attention_scores)
        attention_scores = self.softmax(attention_scores)
        attention_scores = F.cast(self.dropout(attention_scores), value_layer.dtype)
        context_layer = self.matmul(attention_scores.view(-1, attention_scores.shape[-2], attention_scores.shape[-1]), value_layer)
        context_layer = context_layer.view(-1, self.num_attention_heads, context_layer.shape[-2], context_layer.shape[-1]).transpose(self.trans_shape)
        new_context_layer_shape = context_layer.shape[:-2] + (-1,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer


    def disentangled_attention_bias(self, query_layer, key_layer, relative_pos, rel_embeddings, scale_factor):
        if relative_pos is None:
            q = query_layer.shape[-2]
            relative_pos = build_relative_position(q, key_layer.shape[-2], 
                                                    bucket_size = self.position_buckets, 
                                                    max_position = self.max_relative_positions)
        if relative_pos.ndim == 2:
            relative_pos = relative_pos.view((1, 1, relative_pos.shape[0], relative_pos.shape[1]))
        elif relative_pos.ndim == 3:
            relative_pos = relative_pos.view((relative_pos.shape[0], 1, relative_pos.shape[1], relative_pos.shape[2]))
        # bxhxqxk
        elif relative_pos.ndim != 4:
            raise ValueError(f'Relative postion ids must be of dim 2 or 3 or 4. {relative_pos.ndim}')

        att_span = self.pos_ebd_size

        rel_embeddings = F.expand_dims(rel_embeddings[self.pos_ebd_size - att_span: self.pos_ebd_size + att_span, :], 0)#.repeat(query_layer.shape[0]//self.num_attention_heads, 1, 1)
        if self.share_att_key:
            pos_query_layer = self.transpose_for_scores(rel_embeddings, 
                                                        self.query_proj, 
                                                        self.num_attention_heads, 
                                                        self.shape_to_2d, 
                                                        rel_embeddings.shape[1])
            pos_query_layer = pos_query_layer.repeat(query_layer.shape[0] // self.num_attention_heads, axis=0) 
            pos_key_layer = self.transpose_for_scores(rel_embeddings, 
                                                      self.key_proj, 
                                                      self.num_attention_heads, 
                                                      self.shape_to_2d, 
                                                      rel_embeddings.shape[1])
            pos_key_layer = pos_key_layer.repeat(query_layer.shape[0] // self.num_attention_heads, axis=0) 
            
            # pos_query_layer = self.transpose_for_scores(self.query_proj(rel_embeddings), self.num_attention_heads)\
            #     .repeat(query_layer.shape[0]//self.num_attention_heads, 1, 1) #.split(self.all_head_size, dim=-1)
            # pos_key_layer = self.transpose_for_scores(self.key_proj(rel_embeddings), self.num_attention_heads)\
            #     .repeat(query_layer.shape[0]//self.num_attention_heads, 1, 1) #.split(self.all_head_size, dim=-1)
        else:
            if 'c2p' in self.pos_att_type or 'p2p' in self.pos_att_type:
                pos_key_layer = self.transpose_for_scores(rel_embeddings, 
                                                          self.pos_key_proj, 
                                                          self.num_attention_heads, 
                                                          self.shape_to_2d, 
                                                          rel_embeddings.shape[1])
                pos_key_layer = pos_key_layer.repeat(query_layer.shape[0]//self.num_attention_heads, axis=0) 

                # pos_key_layer = self.transpose_for_scores(self.pos_key_proj(rel_embeddings), self.num_attention_heads)\
                #     .repeat(query_layer.shape[0]//self.num_attention_heads, 1, 1) #.split(self.all_head_size, dim=-1)
            if 'p2c' in self.pos_att_type or 'p2p' in self.pos_att_type:
                pos_query_layer = self.transpose_for_scores(rel_embeddings, 
                                                            self.pos_query_proj, 
                                                            self.num_attention_heads, 
                                                            self.shape_to_2d, 
                                                            rel_embeddings.shape[1])
                pos_query_layer = pos_query_layer.repeat(query_layer.shape[0]//self.num_attention_heads, axis=0) 
                
                # pos_query_layer = self.transpose_for_scores(self.pos_query_proj(rel_embeddings), self.num_attention_heads)\
                #     .repeat(query_layer.shape[0]//self.num_attention_heads, 1, 1) #.split(self.all_head_size, dim=-1)

        score = 0
        # content->position
        if 'c2p' in self.pos_att_type:
            scale = 1 / math.sqrt(pos_key_layer.shape[-1] * scale_factor)
            c2p_att = self.matmul(query_layer, pos_key_layer.transpose(0, 2, 1).astype(query_layer.dtype) * scale)
            c2p_pos = O.clip_by_value(relative_pos + att_span, 0, att_span * 2 - 1)
            index = O.BroadcastTo((query_layer.shape[0], query_layer.shape[1], relative_pos.shape[-1]))(c2p_pos.squeeze(0))
            c2p_att = self.gather(c2p_att, -1, index)
            score += c2p_att

        # position->content
        if 'p2c' in self.pos_att_type or 'p2p' in self.pos_att_type:
            scale = 1 / math.sqrt(pos_query_layer.shape[-1]*scale_factor)
            if key_layer.shape[-2] != query_layer.shape[-2]:
                r_pos = build_relative_position(key_layer.shape[-2], key_layer.shape[-2], 
                                                bucket_size = self.position_buckets, 
                                                max_position = self.max_relative_positions)
                r_pos = F.expand_dims(r_pos, 0)
            else:
                r_pos = relative_pos

            p2c_pos = O.clip_by_value(-r_pos + att_span, 0, att_span * 2 - 1)
            if query_layer.shape[-2] != key_layer.shape[-2]:
                pos_index = F.expand_dims(relative_pos[:, :, :, 0], -1)

        if 'p2c' in self.pos_att_type:
            p2c_att = self.matmul(key_layer, pos_query_layer.transpose(0, 2, 1).astype(key_layer.dtype) * scale)
            index = O.BroadcastTo((query_layer.shape[0], key_layer.shape[-2], key_layer.shape[-2]))(p2c_pos.squeeze(0))
            p2c_att = self.gather(p2c_att, -1, index).transpose(0, 2, 1)
            if query_layer.shape[-2] != key_layer.shape[-2]:
                index = O.BroadcastTo(p2c_att.shape[:2] + (pos_index.shape[-2], key_layer.shape[-2]))(p2c_att)
                p2c_att = self.gather(p2c_att, -2, index)
            score += p2c_att

        # position->position
        if 'p2p' in self.pos_att_type:
            pos_query = pos_query_layer[:, :, att_span:, :]
            p2p_att = self.matmul(pos_query, pos_key_layer.transpose(0, 2, 1))
            p2p_att = O.BroadcastTo(query_layer.shape[:2] + p2p_att.shape[2:])(p2p_att)
            if query_layer.shape[-2] != key_layer.shape[-2]:
                index = O.BroadcastTo(query_layer.shape[:2] + (pos_index.shape[-2], p2p_att.shape[-1]))(pos_index)
                p2p_att = self.gather(p2p_att, -2, index)
            p2p_att = self.gather(p2p_att, -1, O.BroadcastTo((query_layer.shape[0], query_layer.shape[1], query_layer.shape[2], relative_pos.shape[-1])))(c2p_pos)
            score += p2p_att

        return score


class BertAttention(nn.Cell):
    """
    Apply multi-headed attention from "from_tensor" to "to_tensor".

    Args:
        from_tensor_width (int): Size of last dim of from_tensor.
        to_tensor_width (int): Size of last dim of to_tensor.
        num_attention_heads (int): Number of attention heads. Default: 1.
        size_per_head (int): Size of each attention head. Default: 512.
        query_act (str): Activation function for the query transform. Default: None.
        key_act (str): Activation function for the key transform. Default: None.
        value_act (str): Activation function for the value transform. Default: None.
        has_attention_mask (bool): Specifies whether to use attention mask. Default: False.
        attention_probs_dropout_prob (float): The dropout probability for
                                      BertAttention. Default: 0.0.
        use_one_hot_embeddings (bool): Specifies whether to use one hot encoding form. Default: False.
        initializer_range (float): Initialization value of TruncatedNormal. Default: 0.02.
        use_relative_positions (bool): Specifies whether to use relative positions. Default: False.
        compute_type (:class:`mindspore.dtype`): Compute type in BertAttention. Default: mstype.float32.
    """
    def __init__(self,
                 from_tensor_width,
                 to_tensor_width,
                 num_attention_heads=1,
                 size_per_head=512,
                 query_act=None,
                 key_act=None,
                 value_act=None,
                 has_attention_mask=False,
                 attention_probs_dropout_prob=0.0,
                 use_one_hot_embeddings=False,
                 initializer_range=0.02,
                 use_relative_positions=False,
                 compute_type=mstype.float32):

        super(BertAttention, self).__init__()
        self.num_attention_heads = num_attention_heads
        self.size_per_head = size_per_head
        self.has_attention_mask = has_attention_mask
        self.use_relative_positions = use_relative_positions

        self.scores_mul = 1.0 / math.sqrt(float(self.size_per_head))
        self.reshape = P.Reshape()
        self.shape_from_2d = (-1, from_tensor_width)
        self.shape_to_2d = (-1, to_tensor_width)
        weight = TruncatedNormal(initializer_range)
        units = num_attention_heads * size_per_head
        self.query = nn.Dense(from_tensor_width,
                                    units,
                                    activation=query_act,
                                    weight_init=weight).to_float(compute_type)
        self.key = nn.Dense(to_tensor_width,
                                  units,
                                  activation=key_act,
                                  weight_init=weight).to_float(compute_type)
        self.value = nn.Dense(to_tensor_width,
                                    units,
                                    activation=value_act,
                                    weight_init=weight).to_float(compute_type)

        self.matmul_trans_b = P.BatchMatMul(transpose_b=True)
        self.multiply = P.Mul()
        self.transpose = P.Transpose()
        self.trans_shape = (0, 2, 1, 3)
        self.trans_shape_relative = (2, 0, 1, 3)
        self.trans_shape_position = (1, 2, 0, 3)
        self.multiply_data = -10000.0
        self.matmul = P.BatchMatMul()

        self.softmax = nn.Softmax()
        self.dropout = nn.Dropout(attention_probs_dropout_prob)

        if self.has_attention_mask:
            self.expand_dims = P.ExpandDims()
            self.sub = P.Sub()
            self.add = P.Add()
            self.cast = P.Cast()
            self.get_dtype = P.DType()

        self.shape_return = (-1, num_attention_heads * size_per_head)

        self.cast_compute_type = SaturateCast(dst_type=compute_type)
        if self.use_relative_positions:
            self._generate_relative_positions_embeddings = \
                RelaPosEmbeddingsGenerator(depth=size_per_head,
                                           max_relative_position=16,
                                           initializer_range=initializer_range,
                                           use_one_hot_embeddings=use_one_hot_embeddings)

    def construct(self, to_tensor, from_tensor, attention_mask, **kwargs):
        """reshape 2d/3d input tensors to 2d"""
        shape_from = F.shape(attention_mask)[2]
        from_tensor = F.depend(from_tensor, shape_from)
        from_tensor_2d = self.reshape(from_tensor, self.shape_from_2d)
        to_tensor_2d = self.reshape(to_tensor, self.shape_to_2d)
        query_out = self.query(from_tensor_2d)
        key_out = self.key(to_tensor_2d)
        value_out = self.value(to_tensor_2d)

        query_layer = self.reshape(query_out, (-1, shape_from, self.num_attention_heads, self.size_per_head))
        query_layer = self.transpose(query_layer, self.trans_shape)
        key_layer = self.reshape(key_out, (-1, shape_from, self.num_attention_heads, self.size_per_head))
        key_layer = self.transpose(key_layer, self.trans_shape)

        attention_scores = self.matmul_trans_b(query_layer, key_layer)

        # use_relative_position, supplementary logic
        if self.use_relative_positions:
            # relations_keys is [F|T, F|T, H]
            relations_keys = self._generate_relative_positions_embeddings(shape_from)
            relations_keys = self.cast_compute_type(relations_keys)
            # query_layer_t is [F, B, N, H]
            query_layer_t = self.transpose(query_layer, self.trans_shape_relative)
            # query_layer_r is [F, B * N, H]
            query_layer_r = self.reshape(query_layer_t,
                                         (shape_from,
                                          -1,
                                          self.size_per_head))
            # key_position_scores is [F, B * N, F|T]
            key_position_scores = self.matmul_trans_b(query_layer_r,
                                                      relations_keys)
            # key_position_scores_r is [F, B, N, F|T]
            key_position_scores_r = self.reshape(key_position_scores,
                                                 (shape_from,
                                                  -1,
                                                  self.num_attention_heads,
                                                  shape_from))
            # key_position_scores_r_t is [B, N, F, F|T]
            key_position_scores_r_t = self.transpose(key_position_scores_r,
                                                     self.trans_shape_position)
            attention_scores = attention_scores + key_position_scores_r_t

        attention_scores = self.multiply(self.scores_mul, attention_scores)

        if self.has_attention_mask:
            attention_mask = self.expand_dims(attention_mask, 1)
            multiply_out = self.sub(self.cast(F.tuple_to_array((1.0,)), self.get_dtype(attention_scores)),
                                    self.cast(attention_mask, self.get_dtype(attention_scores)))

            adder = self.multiply(multiply_out, self.multiply_data)
            attention_scores = self.add(adder, attention_scores)

        attention_probs = self.softmax(attention_scores)
        attention_probs = self.dropout(attention_probs)

        value_layer = self.reshape(value_out, (-1, shape_from, self.num_attention_heads, self.size_per_head))
        value_layer = self.transpose(value_layer, self.trans_shape)
        context_layer = self.matmul(attention_probs, value_layer)

        # use_relative_position, supplementary logic
        if self.use_relative_positions:
            # relations_values is [F|T, F|T, H]
            relations_values = self._generate_relative_positions_embeddings(shape_from)
            relations_values = self.cast_compute_type(relations_values)
            # attention_probs_t is [F, B, N, T]
            attention_probs_t = self.transpose(attention_probs, self.trans_shape_relative)
            # attention_probs_r is [F, B * N, T]
            attention_probs_r = self.reshape(
                attention_probs_t,
                (shape_from,
                 -1,
                 shape_from))
            # value_position_scores is [F, B * N, H]
            value_position_scores = self.matmul(attention_probs_r,
                                                relations_values)
            # value_position_scores_r is [F, B, N, H]
            value_position_scores_r = self.reshape(value_position_scores,
                                                   (shape_from,
                                                    -1,
                                                    self.num_attention_heads,
                                                    self.size_per_head))
            # value_position_scores_r_t is [B, N, F, H]
            value_position_scores_r_t = self.transpose(value_position_scores_r,
                                                       self.trans_shape_position)
            context_layer = context_layer + value_position_scores_r_t

        context_layer = self.transpose(context_layer, self.trans_shape)
        context_layer = self.reshape(context_layer, context_layer.shape[:2] + (-1,))

        return context_layer


class Attention(nn.Cell):
    """
    Apply self-attention.

    Args:
        hidden_size (int): Size of the bert encoder layers.
        num_attention_heads (int): Number of attention heads. Default: 12.
        attention_probs_dropout_prob (float): The dropout probability for
                                      BertAttention. Default: 0.1.
        use_one_hot_embeddings (bool): Specifies whether to use one_hot encoding form. Default: False.
        initializer_range (float): Initialization value of TruncatedNormal. Default: 0.02.
        hidden_dropout_prob (float): The dropout probability for BertOutput. Default: 0.1.
        use_relative_positions (bool): Specifies whether to use relative positions. Default: False.
        compute_type (:class:`mindspore.dtype`): Compute type in BertSelfAttention. Default: mstype.float32.
    """
    def __init__(self,
                 hidden_size,
                 attention=BertAttention,
                 num_attention_heads=12,
                 attention_probs_dropout_prob=0.1,
                 use_one_hot_embeddings=False,
                 initializer_range=0.02,
                 hidden_dropout_prob=0.1,
                 use_relative_positions=False,
                 compute_type=mstype.float32,
                 **kwargs):
        super(Attention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError("The hidden size (%d) is not a multiple of the number "
                             "of attention heads (%d)" % (hidden_size, num_attention_heads))

        self.size_per_head = int(hidden_size / num_attention_heads)
        # if config is not None:
        #     self.self = DisentangledSelfAttention(config=config, compute_type=compute_type)
        # else:
        #     self.self = BertAttention(
        #         from_tensor_width=hidden_size,
        #         to_tensor_width=hidden_size,
        #         num_attention_heads=num_attention_heads,
        #         size_per_head=self.size_per_head,
        #         attention_probs_dropout_prob=attention_probs_dropout_prob,
        #         use_one_hot_embeddings=use_one_hot_embeddings,
        #         initializer_range=initializer_range,
        #         use_relative_positions=use_relative_positions,
        #         has_attention_mask=True,
        #         compute_type=compute_type)
        if isinstance(attention, type):
            self.attention = attention(
                                from_tensor_width=hidden_size,
                                to_tensor_width=hidden_size,
                                num_attention_heads=num_attention_heads,
                                size_per_head=self.size_per_head,
                                attention_probs_dropout_prob=attention_probs_dropout_prob,
                                use_one_hot_embeddings=use_one_hot_embeddings,
                                initializer_range=initializer_range,
                                use_relative_positions=use_relative_positions,
                                has_attention_mask=True,
                                compute_type=compute_type)
        else:
            self.attention = copy.deepcopy(attention)

        self.output = BertOutput(in_channels=hidden_size,
                                 out_channels=hidden_size,
                                 initializer_range=initializer_range,
                                 dropout_prob=hidden_dropout_prob,
                                 compute_type=compute_type)
        self.reshape = P.Reshape()
        self.shape = (-1, hidden_size)

    def construct(self, input_tensor, query_states, attention_mask, **kwargs):
        attention_output = self.attention(input_tensor, query_states, attention_mask, **kwargs)
        output = self.output(attention_output, input_tensor)
        return output


class BertOutput(nn.Cell):
    """
    Apply a linear computation to hidden status and a residual computation to input.

    Args:
        in_channels (int): Input channels.
        out_channels (int): Output channels.
        initializer_range (float): Initialization value of TruncatedNormal. Default: 0.02.
        dropout_prob (float): The dropout probability. Default: 0.1.
        compute_type (:class:`mindspore.dtype`): Compute type in BertTransformer. Default: mstype.float32.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 initializer_range=0.02,
                 dropout_prob=0.1,
                 compute_type=mstype.float32):
        super(BertOutput, self).__init__()
        self.dense = nn.Dense(in_channels, out_channels,
                              weight_init=TruncatedNormal(initializer_range)).to_float(compute_type)
        self.dropout = nn.Dropout(dropout_prob)
        self.dropout_prob = dropout_prob
        self.add = P.Add()
        self.LayerNorm = nn.LayerNorm((out_channels,)).to_float(compute_type)
        self.cast = P.Cast()

    def construct(self, hidden_status, input_tensor):
        output = self.dense(hidden_status)
        output = self.dropout(output)
        output = self.add(input_tensor, output)
        output = self.LayerNorm(output)
        return output


class RelaPosMatrixGenerator(nn.Cell):
    """
    Generates matrix of relative positions between inputs.

    Args:
        length (int): Length of one dim for the matrix to be generated.
        max_relative_position (int): Max value of relative position.
    """
    def __init__(self, max_relative_position):
        super(RelaPosMatrixGenerator, self).__init__()
        self._max_relative_position = max_relative_position
        self._min_relative_position = -max_relative_position

        self.tile = P.Tile()
        self.range_mat = P.Reshape()
        self.sub = P.Sub()
        self.expanddims = P.ExpandDims()
        self.cast = P.Cast()

    def construct(self, length):
        """Generates matrix of relative positions between inputs."""
        range_vec_row_out = self.cast(F.tuple_to_array(F.make_range(length)), mstype.int32)
        range_vec_col_out = self.range_mat(range_vec_row_out, (length, -1))
        tile_row_out = self.tile(range_vec_row_out, (length,))
        tile_col_out = self.tile(range_vec_col_out, (1, length))
        range_mat_out = self.range_mat(tile_row_out, (length, length))
        transpose_out = self.range_mat(tile_col_out, (length, length))
        distance_mat = self.sub(range_mat_out, transpose_out)

        distance_mat_clipped = C.clip_by_value(distance_mat,
                                               self._min_relative_position,
                                               self._max_relative_position)

        # Shift values to be >=0. Each integer still uniquely identifies a
        # relative position difference.
        final_mat = distance_mat_clipped + self._max_relative_position
        return final_mat


class RelaPosEmbeddingsGenerator(nn.Cell):
    """
    Generates tensor of size [length, length, depth].

    Args:
        length (int): Length of one dim for the matrix to be generated.
        depth (int): Size of each attention head.
        max_relative_position (int): Maxmum value of relative position.
        initializer_range (float): Initialization value of TruncatedNormal.
        use_one_hot_embeddings (bool): Specifies whether to use one hot encoding form. Default: False.
    """
    def __init__(self,
                 depth,
                 max_relative_position,
                 initializer_range,
                 use_one_hot_embeddings=False):
        super(RelaPosEmbeddingsGenerator, self).__init__()
        self.depth = depth
        self.vocab_size = max_relative_position * 2 + 1
        self.use_one_hot_embeddings = use_one_hot_embeddings

        self.embeddings_table = Parameter(
            initializer(TruncatedNormal(initializer_range),
                        [self.vocab_size, self.depth]))

        self.relative_positions_matrix = RelaPosMatrixGenerator(max_relative_position=max_relative_position)
        self.reshape = P.Reshape()
        self.one_hot = nn.OneHot(depth=self.vocab_size)
        self.shape = P.Shape()
        self.gather = P.Gather()  # index_select
        self.matmul = P.BatchMatMul()

    def construct(self, length):
        """Generate embedding for each relative position of dimension depth."""
        relative_positions_matrix_out = self.relative_positions_matrix(length)

        if self.use_one_hot_embeddings:
            flat_relative_positions_matrix = self.reshape(relative_positions_matrix_out, (-1,))
            one_hot_relative_positions_matrix = self.one_hot(
                flat_relative_positions_matrix)
            embeddings = self.matmul(one_hot_relative_positions_matrix, self.embeddings_table)
            my_shape = self.shape(relative_positions_matrix_out) + (self.depth,)
            embeddings = self.reshape(embeddings, my_shape)
        else:
            embeddings = self.gather(self.embeddings_table,
                                     relative_positions_matrix_out, 0)
        return embeddings


class EncoderLayer(nn.Cell):
    """
    Encoder cells used in BertTransformer.

    Args:
        hidden_size (int): Size of the bert encoder layers. Default: 768.
        num_attention_heads (int): Number of attention heads. Default: 12.
        intermediate_size (int): Size of intermediate layer. Default: 3072.
        attention_probs_dropout_prob (float): The dropout probability for
                                      BertAttention. Default: 0.02.
        use_one_hot_embeddings (bool): Specifies whether to use one hot encoding form. Default: False.
        initializer_range (float): Initialization value of TruncatedNormal. Default: 0.02.
        hidden_dropout_prob (float): The dropout probability for BertOutput. Default: 0.1.
        use_relative_positions (bool): Specifies whether to use relative positions. Default: False.
        hidden_act (str): Activation function. Default: "gelu".
        compute_type (:class:`mindspore.dtype`): Compute type in attention. Default: mstype.float32.
    """
    def __init__(self,
                 hidden_size=768,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 attention_probs_dropout_prob=0.02,
                 use_one_hot_embeddings=False,
                 initializer_range=0.02,
                 hidden_dropout_prob=0.1,
                 use_relative_positions=False,
                 hidden_act="gelu",
                 attension=BertAttention,
                 compute_type=mstype.float32):
        super(EncoderLayer, self).__init__()
        self.attention = Attention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            use_one_hot_embeddings=use_one_hot_embeddings,
            initializer_range=initializer_range,
            hidden_dropout_prob=hidden_dropout_prob,
            use_relative_positions=use_relative_positions,
            attention=attension,
            compute_type=compute_type)
        self.intermediate = nn.Dense(in_channels=hidden_size,
                                     out_channels=intermediate_size,
                                     activation=hidden_act,
                                     weight_init=TruncatedNormal(initializer_range)).to_float(compute_type)
        self.output = BertOutput(in_channels=intermediate_size,
                                 out_channels=hidden_size,
                                 initializer_range=initializer_range,
                                 dropout_prob=hidden_dropout_prob,
                                 compute_type=compute_type)

    def construct(self, hidden_states, query_states, attention_mask, **kwargs):
        # self-attention
        attention_output = self.attention(hidden_states, query_states, attention_mask, **kwargs)
        # feed construct
        intermediate_output = self.intermediate(attention_output)
        # add and normalize
        output = self.output(intermediate_output, attention_output)
        return output


class ConvLayer(nn.Cell):
    def __init__(self, conv_kernel_size=3, conv_groups=1, conv_act='tanh', 
                 hidden_size=768, hidden_dropout_prob=0.1, compute_type=mstype.float32, **kwargs):
        super().__init__()
        self.conv_act = nn.get_activation(conv_act)
        self.conv = nn.Conv1d(hidden_size, hidden_size, conv_kernel_size, group=conv_groups).to_float(compute_type)
        self.LayerNorm = nn.LayerNorm((hidden_size,), epsilon=kwargs.get('layer_norm_eps', 1e-7))
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def construct(self, hidden_states, residual_states, input_mask):
        out = self.conv(hidden_states.transpose(0, 2, 1)).transpose(0, 2, 1)
        rmask = 1 - input_mask
        out = P.MaskedFill()(out, O.BroadcastTo(out.shape)(F.expand_dims(rmask, -1)).astype(mstype.bool_), 0.0)
        out = self.conv_act(self.dropout(out))
        output_states = self.LayerNorm(residual_states + out)
        return output_states


class SaturateCast(nn.Cell):
    """
    Performs a safe saturating cast. This operation applies proper clamping before casting to prevent
    the danger that the value will overflow or underflow.

    Args:
        src_type (:class:`mindspore.dtype`): The type of the elements of the input tensor. Default: mstype.float32.
        dst_type (:class:`mindspore.dtype`): The type of the elements of the output tensor. Default: mstype.float32.
    """
    def __init__(self, src_type=mstype.float32, dst_type=mstype.float32):
        super(SaturateCast, self).__init__()
        np_type = mstype.dtype_to_nptype(dst_type)

        self.tensor_min_type = float(numpy.finfo(np_type).min)
        self.tensor_max_type = float(numpy.finfo(np_type).max)

        self.min_op = P.Minimum()
        self.max_op = P.Maximum()
        self.cast = P.Cast()
        self.dst_type = dst_type

    def construct(self, x):
        out = self.max_op(x, self.tensor_min_type)
        out = self.min_op(out, self.tensor_max_type)
        return self.cast(out, self.dst_type)


class Encoder(nn.Cell):
    """
    Multi-layer bert transformer.

    Args:
        hidden_size (int): Size of the encoder layers.
        num_hidden_layers (int): Number of hidden layers in encoder cells.
        num_attention_heads (int): Number of attention heads in encoder cells. Default: 12.
        intermediate_size (int): Size of intermediate layer in encoder cells. Default: 3072.
        attention_probs_dropout_prob (float): The dropout probability for
                                      BertAttention. Default: 0.1.
        use_one_hot_embeddings (bool): Specifies whether to use one hot encoding form. Default: False.
        initializer_range (float): Initialization value of TruncatedNormal. Default: 0.02.
        hidden_dropout_prob (float): The dropout probability for BertOutput. Default: 0.1.
        use_relative_positions (bool): Specifies whether to use relative positions. Default: False.
        hidden_act (str): Activation function used in the encoder cells. Default: "gelu".
        compute_type (:class:`mindspore.dtype`): Compute type in BertTransformer. Default: mstype.float32.
        return_all_encoders (bool): Specifies whether to return all encoders. Default: False.
    """
    def __init__(self,
                 hidden_size,
                 num_hidden_layers,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 attention_probs_dropout_prob=0.1,
                 use_one_hot_embeddings=False,
                 initializer_range=0.02,
                 hidden_dropout_prob=0.1,
                 use_relative_positions=False,
                 hidden_act="gelu",
                 attention=BertAttention,
                 compute_type=mstype.float32,
                 return_all_encoders=False,
                 max_position_embeddings=512,
                 **kwargs):
        super(Encoder, self).__init__()
        self.return_all_encoders = return_all_encoders
        if 'config' in kwargs:
            kwargs.update(kwargs['config'].to_dict())
        layers = []
        self.num_hidden_layers = num_hidden_layers
        for _ in range(num_hidden_layers):
            layer = EncoderLayer(hidden_size=hidden_size,
                                 num_attention_heads=num_attention_heads,
                                 intermediate_size=intermediate_size,
                                 attention_probs_dropout_prob=attention_probs_dropout_prob,
                                 use_one_hot_embeddings=use_one_hot_embeddings,
                                 initializer_range=initializer_range,
                                 hidden_dropout_prob=hidden_dropout_prob,
                                 use_relative_positions=use_relative_positions,
                                 hidden_act=hidden_act,
                                 attension=attention,
                                 compute_type=compute_type)
            layers.append(layer)

        self.layer = nn.CellList(layers)
        self.relative_attention = use_relative_positions
        if use_relative_positions:
            self.position_buckets = kwargs.get('position_buckets', -1)
            self.max_relative_positions = kwargs.get('max_relative_positions', -1)
            if self.max_relative_positions < 1:
                    self.max_relative_positions = max_position_embeddings
            self.create_rel_embeddings(self.max_relative_positions, position_buckets=self.position_buckets,
                                        hidden_size=hidden_size)
        else:
            self.rel_embeddings = None

        self.norm_rel_ebd = [x.strip() for x in kwargs.get('norm_rel_ebd', 'none').lower().split('|')]
        if 'layer_norm' in self.norm_rel_ebd:
            self.LayerNorm = nn.LayerNorm((hidden_size,), epsilon=kwargs.get('layer_norm_eps', 1e-7))
        else:
            self.LayerNorm = None
        
        # add 1-D conv to improve aggregation
        kernel_size = kwargs.get('conv_kernel_size', 0)
        self.with_conv = False
        if kernel_size > 0:
            self.with_conv = True
            self.conv = ConvLayer(conv_kernel_size=kernel_size, hidden_size=hidden_size, 
                                  hidden_dropout_prob=hidden_dropout_prob, compute_type=compute_type)
        self.reshape = P.Reshape()
        self.shape = (-1, hidden_size)


    def create_rel_embeddings(self, max_relative_positions, position_buckets=512, hidden_size=768, **kwargs):
        pos_ebd_size = max_relative_positions * 2
        if position_buckets > 0:
                pos_ebd_size = position_buckets * 2
        self.rel_embeddings = nn.Embedding(pos_ebd_size, hidden_size)

    def construct(self, hidden_states, attention_mask, output_all_encoded_layers=True, 
                    return_att=False, query_states = None, relative_pos=None):
        """Multi-layer bert transformer."""
        # prev_output = self.reshape(hidden_states, self.shape)
        all_encoder_layers = []
        if attention_mask.ndim <= 2:
            input_mask = attention_mask
        else:
            input_mask = (attention_mask.sum(-2) > 0)
        if self.relative_attention:
            relative_pos = get_rel_pos(hidden_states, query_states, relative_pos, 
                                        relative_attention=self.relative_attention,
                                        position_buckets=self.position_buckets,
                                        max_relative_positions=self.max_relative_positions)
        # all_encoder_layers = []
        att_matrices = []
        if isinstance(hidden_states, Tensor):
            next_kv = hidden_states
        else:
            next_kv = hidden_states[0]
            
        if self.relative_attention is not None:
            rel_embeddings = get_rel_embedding(self.rel_embeddings.embedding_table, self.relative_attention, self.norm_rel_ebd, self.LayerNorm)
        else:
            rel_embeddings = None

        output_states = hidden_states
        att_m = None
        for i, layer_module in enumerate(self.layer):
            output_states = layer_module(next_kv, query_states, attention_mask, return_att=return_att, 
                                         relative_pos=relative_pos, rel_embeddings=rel_embeddings)

            if return_att:
                output_states, att_m = output_states
            else:
                att_m = None

            if i == 0 and self.with_conv:
                prenorm = output_states #output['prenorm_states']
                output_states = self.conv(hidden_states, prenorm, input_mask)

            if query_states is not None:
                query_states = output_states
                if isinstance(hidden_states, Sequence):
                    next_kv = hidden_states[i + 1] if i + 1 < len(self.layer) else None
            else:
                next_kv = output_states

            if output_all_encoded_layers:
                all_encoder_layers.append(output_states)
                if return_att:
                    att_matrices.append(att_m)
            else:
                if i == self.num_hidden_layers - 1:
                    all_encoder_layers.append(output_states)
                if return_att:
                    att_matrices.append(att_m)
        # if not output_all_encoded_layers:
        #     all_encoder_layers.append(output_states)
        #     if return_att:
        #         att_matrices.append(att_m)
        return {
            'hidden_states': all_encoder_layers,
            'attention_matrices': att_matrices
            }


        #     layer_output = layer_module(prev_output, attention_mask)
        #     prev_output = layer_output

        #     if self.return_all_encoders:
        #         shape = F.shape(hidden_states)
        #         layer_output = self.reshape(layer_output, shape)
        #         all_encoder_layers = all_encoder_layers + (layer_output,)

        # if not self.return_all_encoders:
        #     shape = F.shape(hidden_states)
        #     prev_output = self.reshape(prev_output, shape)
        #     all_encoder_layers = all_encoder_layers + (prev_output,)
        # return all_encoder_layers


class Pooler(nn.Cell):
    def __init__(self, hidden_size, initializer_range, compute_type='float16') -> None:
        super().__init__()
        weight_init = TruncatedNormal(initializer_range)
        self.dense = nn.Dense(hidden_size,
                              hidden_size,
                              weight_init=weight_init,
                              activation="tanh").to_float(compute_type)

    def construct(self, hidden_states):
        return self.dense(hidden_states)

class GetMaskedLMOutput(nn.Cell):
    """
    Get masked lm output.

    Args:
        config (BertConfig): The config of BertModel.

    Returns:
        Tensor, masked lm output.
    """

    def __init__(self, **config):
        super(GetMaskedLMOutput, self).__init__()
        self.width = config.get('hidden_size')
        self.reshape = P.Reshape()
        self.gather = P.Gather()

        weight_init = TruncatedNormal(config.get('initializer_range'))
        self.dense = nn.Dense(self.width,
                              config.get('hidden_size'),
                              weight_init=weight_init,
                              activation=config.get('hidden_act')).to_float(config.get('compute_type'))
        self.LayerNorm = nn.LayerNorm((config.get('hidden_size'),)).to_float(config.get('compute_type'))
        self.bias = Parameter(
            initializer(
                'zero',
                config.get('vocab_size')))
        self.matmul = P.MatMul(transpose_b=True)
        self.log_softmax = nn.LogSoftmax(axis=-1)
        self.shape_flat_offsets = (-1, 1)
        self.last_idx = (-1,)
        self.shape_flat_sequence_tensor = (-1, self.width)
        self.cast = P.Cast()
        self.compute_type = config.get('compute_type')
        self.dtype = config.get('compute_type')

    def construct(self,
                  input_tensor,
                  output_weights,
                  positions):
        """Get output log_probs"""
        input_shape = P.Shape()(input_tensor)
        rng = F.tuple_to_array(F.make_range(input_shape[0]))
        flat_offsets = self.reshape(rng * input_shape[1], self.shape_flat_offsets)
        flat_position = self.reshape(positions + flat_offsets, self.last_idx).astype(mstype.int32)
        flat_sequence_tensor = self.reshape(input_tensor, self.shape_flat_sequence_tensor)
        input_tensor = self.gather(flat_sequence_tensor, flat_position, 0)
        input_tensor = self.cast(input_tensor, self.compute_type)
        output_weights = self.cast(output_weights, self.compute_type)
        input_tensor = self.dense(input_tensor)
        input_tensor = self.LayerNorm(input_tensor)
        logits = self.matmul(input_tensor, output_weights)
        logits = self.cast(logits, self.dtype)
        logits = logits + self.bias
        log_probs = self.log_softmax(logits)
        return log_probs


class GetNextSentenceOutput(nn.Cell):
    """
    Get next sentence output.

    Args:
        config (BertConfig): The config of Bert.

    Returns:
        Tensor, next sentence output.
    """

    def __init__(self, **config):
        super(GetNextSentenceOutput, self).__init__()
        self.log_softmax = P.LogSoftmax()
        weight_init = TruncatedNormal(config.get('initializer_range'))
        self.dense = nn.Dense(config.get('hidden_size'), 2,
                              weight_init=weight_init, has_bias=True).to_float(config.get('compute_type'))
        self.dtype = config.get('compute_type')
        self.cast = P.Cast()

    def construct(self, input_tensor):
        logits = self.dense(input_tensor)
        logits = self.cast(logits, self.dtype)
        log_prob = self.log_softmax(logits)
        return log_prob


class FakeHead(nn.Cell):
    """
    A fake head which return the last arg

    Args:
        None

    Returns:
        Tensor, last arg.
    """

    def __init__(self):
        super(FakeHead, self).__init__()

    def construct(self, *input):
        return input[-1]


class BertEvalHead(nn.Cell):
    """
    Provide bert pre-training evaluation head.

    Args:
        config (BertConfig): The config of BertModel.

    Returns:
        tuple: Tensor, total loss. Tensor, ppl
    """

    def __init__(self, vocab_size):
        super(BertEvalHead, self).__init__()
        self.vocab_size = vocab_size
        self.onehot = P.OneHot()
        self.on_value = Tensor(1.0, mstype.float32)
        self.off_value = Tensor(0.0, mstype.float32)
        self.reduce_sum = P.ReduceSum()
        self.reduce_mean = P.ReduceMean()
        self.reshape = P.Reshape()
        self.last_idx = (-1,)
        self.neg = P.Neg()
        self.cast = P.Cast()
        self.argmax = P.Argmax()

    def construct(self, *sample):
        """Defines the computation performed.
            sample: ["input_ids", "input_mask", "token_type_id", "next_sentence_labels",
                     "masked_lm_positions", "masked_lm_ids", "masked_lm_weights"] + 
                    [prediction_scores, seq_relationship_score]
        """
        input_ids = sample[0]
        input_mask = sample[1]
        token_type_id = sample[2]
        masked_lm_positions = sample[4]
        masked_lm_ids = sample[5]
        masked_lm_weights = sample[6]
        prediction_scores = sample[7]
        next_sentence_labels = sample[3]
        seq_relationship_score = sample[8]
        label_ids = self.reshape(masked_lm_ids, self.last_idx)
        label_weights = self.cast(self.reshape(masked_lm_weights, self.last_idx), mstype.float32)
        one_hot_labels = self.onehot(label_ids, self.vocab_size, self.on_value, self.off_value)

        per_example_loss = self.neg(self.reduce_sum(prediction_scores * one_hot_labels, self.last_idx))
        numerator = self.reduce_sum(label_weights * per_example_loss, ())
        denominator = self.reduce_sum(label_weights, ()) + self.cast(F.tuple_to_array((1e-5,)), mstype.float32)
        masked_lm_loss = numerator / denominator

        # next_sentence_loss
        labels = self.reshape(next_sentence_labels, self.last_idx)
        if labels.max() >= 0:
            one_hot_labels = self.onehot(labels, 2, self.on_value, self.off_value)
            per_example_loss = self.neg(self.reduce_sum(
                one_hot_labels * seq_relationship_score, self.last_idx))
            next_sentence_loss = self.reduce_mean(per_example_loss, self.last_idx)
        else:
            next_sentence_loss = 0

        # total_loss
        total_loss = masked_lm_loss + next_sentence_loss

        bs, _ = F.shape(input_ids)

        index = self.argmax(prediction_scores)
        index = self.reshape(index, (bs, -1))
        eval_acc = F.equal(index, masked_lm_ids)
        eval_acc = self.cast(eval_acc, mstype.float32)
        real_acc = eval_acc * masked_lm_weights
        acc = real_acc.sum()
        total = masked_lm_weights.astype(mstype.float32).sum()

        return total_loss, acc, total, prediction_scores, masked_lm_ids


class ClsEvalHead(nn.Cell):
    """
    Provide bert pre-training evaluation head.

    Args:
        config (BertConfig): The config of BertModel.

    Returns:
        tuple: Tensor, total loss. Tensor, ppl
    """

    def __init__(self, num_labels, return_all=False, fp16=False, dropout=0.1):
        super(ClsEvalHead, self).__init__()
        self.onehot = P.OneHot()
        self.on_value = Tensor(1.0, mstype.float32)
        self.off_value = Tensor(0.0, mstype.float32)
        self.reduce_sum = P.ReduceSum()
        self.reduce_mean = P.ReduceMean()
        self.reshape = P.Reshape()
        self.last_idx = (-1,)
        self.neg = P.Neg()
        self.cast = P.Cast()
        self.argmax = P.Argmax()
        self.loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

    def construct(self, logits, labels):
        """Defines the computation performed.
        """
        loss = self.loss_fn(logits, labels)
        return (loss, logits, labels)

class EnhancedMaskDecoder(nn.Cell):
  def __init__(self, **kwargs):
    super().__init__()
    self.config = kwargs
    self.position_biased_input = self.config.get('position_biased_input', True)
    # self.lm_head = GetMaskedLMOutput(**kwargs)
    self.loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True)
    self._create_attention_mask_from_input_mask = CreateAttentionMaskFromInputMask()

  def construct(self, ctx_layers, z_states, attention_mask, 
                encoder_layer, embedding_table, relative_attention, 
                norm_rel_ebd, LayerNorm_fn, relative_pos=None):
    # if attention_mask.ndim != 2:
    #   attention_mask = (attention_mask.sum(-2) > 0)
    #   attention_mask = extended_attention_mask * F.expand_dims(extended_attention_mask.squeeze(-2), -1)
    # elif attention_mask.ndim == 3:
    #   attention_mask = F.expand_dims(attention_mask, 1)
    attention_mask = self._create_attention_mask_from_input_mask(attention_mask)
    hidden_states = ctx_layers[-2]
    hidden_states = F.cast(hidden_states, z_states.dtype)
    # layers = [encoder.layer[-1] for _ in range(2)]
    layer = encoder_layer
    if not self.position_biased_input: 
    #   layers = [encoder.layer[-1] for _ in range(2)]
      
      query_states = F.add(z_states, hidden_states) # .astype()
      #   query_states = z_states
      query_mask = attention_mask
      mlm_ctx_layers = []
      rel_embeddings = get_rel_embedding(embedding_table, relative_attention, norm_rel_ebd, LayerNorm_fn)
      #layer_module(next_kv, query_states, attention_mask, return_att=return_att, relative_pos=relative_pos, rel_embeddings=rel_embeddings)
      for i in range(2):
        # TODO: pass relative pos ids
        output = layer(hidden_states, query_states, query_mask, return_att=False, relative_pos=relative_pos, rel_embeddings=rel_embeddings)
        query_states = output
        mlm_ctx_layers.append(query_states)
    else:
      mlm_ctx_layers = [ctx_layers[-1]]
    # mlm_ctx_layers = self.emd_context_layer(ctx_layers, z_states, attention_mask, encoder, relative_pos=relative_pos)
    ctx_layer = mlm_ctx_layers[-1]
    # lm_logits = self.lm_head(ctx_layer, ebd_weight, masked_lm_positions)
    return ctx_layer

  def emd_context_layer(self, encoder_layers, z_states, attention_mask, encoder, relative_pos=None):
    if attention_mask.ndim <= 2:
      extended_attention_mask = attention_mask.view(attention_mask.shape + (1, 1))
      attention_mask = extended_attention_mask * F.expand_dims(extended_attention_mask.squeeze(-2), -1)
    # elif attention_mask.ndim == 3:
    #   attention_mask = F.expand_dims(attention_mask, 1)
    hidden_states = encoder_layers[-2]
    hidden_states = F.cast(hidden_states, z_states.dtype)
    if not self.position_biased_input: 
      layers = [encoder.layer[-1] for _ in range(2)]
      
      query_states = F.add(z_states, hidden_states) # .astype()
      #   query_states = z_states
      query_mask = attention_mask
      outputs = []
      rel_embeddings = get_rel_embedding(encoder.rel_embeddings, encoder.relative_attention, encoder.norm_rel_ebd, encoder.LayerNorm)
      #layer_module(next_kv, query_states, attention_mask, return_att=return_att, relative_pos=relative_pos, rel_embeddings=rel_embeddings)
      for layer in layers:
        # TODO: pass relative pos ids
        output = layer(hidden_states, query_states, query_mask, return_att=False, relative_pos=relative_pos, rel_embeddings=rel_embeddings)
        query_states = output
        outputs.append(query_states)
    else:
      outputs = [encoder_layers[-1]]
    
    return outputs


def make_log_bucket_position(relative_pos, bucket_size, max_position):
    sign = P.Sign()(relative_pos)
    mid = bucket_size // 2
    a = (relative_pos < mid).astype(mstype.int16)
    b = (relative_pos > -mid).astype(mstype.int16)
    abs_pos = np.where(np.bitwise_and(a, b), mid - 1, np.abs(relative_pos)).astype(mstype.float16)
    log_pos = np.ceil(np.log(abs_pos / mid) / np.log((max_position - 1) / mid) * (mid - 1)) + mid
    bucket_pos = np.where(abs_pos <= mid, relative_pos, log_pos * sign).astype(np.int64)
    return bucket_pos

# @lru_cache(maxsize=128)
def build_relative_position(query_size, key_size, bucket_size=-1, max_position=-1):
    q_ids = np.arange(0, query_size)
    k_ids = np.arange(0, key_size)
    rel_pos_ids = q_ids[:, None] - np.tile(k_ids, (q_ids.shape[0],1))
    if bucket_size > 0 and max_position > 0:
        rel_pos_ids = make_log_bucket_position(rel_pos_ids, bucket_size, max_position)
    # rel_pos_ids = Tensor(rel_pos_ids, dtype=mstype.int64)
    rel_pos_ids = rel_pos_ids[:query_size, :]
    rel_pos_ids = F.expand_dims(rel_pos_ids, 0)
    return rel_pos_ids

def get_rel_embedding(rel_embeddings, relative_attention, norm_rel_ebd, layer_norm):
    if relative_attention:
        if 'layer_norm' in norm_rel_ebd:
            rel_embeddings = layer_norm(rel_embeddings)
    return rel_embeddings


def get_rel_pos(hidden_states, query_states=None, relative_pos=None, relative_attention=None, position_buckets=None, max_relative_positions=512):
    if relative_attention and relative_pos is None:
        q = query_states.shape[-2] if query_states is not None else hidden_states.shape[-2]
        relative_pos = build_relative_position(q, hidden_states.shape[-2], bucket_size=position_buckets, max_position=max_relative_positions)
    return relative_pos
