# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# zbo@zju.edu.cn
# 2022-08-08
# ============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .models import *
from .config import ModelConfig

models = {
    'bert': BertModel,
    'bert_pretraining': BertPreTraining,
    'deberta': Deberta,
    'deberta_pretraining': DebertaPreTraining
}


def build_transformer_model(
    config_path=None,
    model='bert',
    application='encoder',
    **kwargs
):
    """根据配置文件构建模型，可选加载checkpoint权重
    """
    compute_type = kwargs.get('compute_type', mstype.float16)
    if config_path is not None:
        configs = ModelConfig.from_json_file(config_path)
    else:
        configs = ModelConfig()
    configs.update(**kwargs)
    # mindspore中dropout是keep rate
    if configs['hidden_dropout_prob'] < 0.5:
        configs['hidden_dropout_prob'] = 1 - configs.get('hidden_dropout_prob')
        configs['attention_probs_dropout_prob'] = 1 - configs.get('attention_probs_dropout_prob')
    if 'max_position' not in configs:
        configs['max_position'] = configs.get('max_position_embeddings', 512)
    if 'dropout_rate' not in configs:
        configs['dropout_rate'] = configs.get('hidden_dropout_prob')
    if 'attention_dropout_rate' not in configs:
        configs['attention_dropout_rate'] = configs.get(
            'attention_probs_dropout_prob'
        )
    if 'segment_vocab_size' not in configs:
        configs['segment_vocab_size'] = configs.get('type_vocab_size', 2)

    if 'compute_type' not in configs:
        configs['compute_type'] = compute_type

    if isinstance(model, str):
        model = model.lower()
        MODEL = models[model]
    else:
        MODEL = model

    # application = application.lower()
    # if application in ['lm', 'unilm'] and model in ['electra', 't5']:
    #     raise ValueError(
    #         '"%s" model can not be used as "%s" application.\n' %
    #         (model, application)
    #     )

    # if application == 'lm':
    #     MODEL = extend_with_language_model(MODEL)
    # elif application == 'unilm':
    #     MODEL = extend_with_unified_language_model(MODEL)
    transformer = MODEL(**configs.to_dict(), config=configs)

    # if checkpoint_path is not None:
    #     transformer.load_weights_from_checkpoint(checkpoint_path)

    return transformer, configs