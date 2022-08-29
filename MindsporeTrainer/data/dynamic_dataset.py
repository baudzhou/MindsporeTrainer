# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# zbo@zju.edu.cn
# 2022-08-08
# ============================================================================

import random
import mmap
import numpy as np
from bisect import bisect
import loguru
import numpy as np
import mindspore.dataset as ds
import mindspore.dataset.transforms as C
import mindspore.common.dtype as mstype
logger=loguru.logger

def create_dynamic_dataset(examples, feature_fn, batch_size, output_columns, column_names=["example", "label"],
                           buffer_size=1000, repeat=1, num_workers=1, num_shards=None,
                           shard_id=None):
    '''
        Provide dynamic data set.
        Args:
            examples: text example list
            feature_fn: function for generating mask etc.
    '''
    dataset = ds.GeneratorDataset(examples, 
                        shuffle=True,
                        column_names=column_names,
                        num_shards=num_shards,
                        shard_id=shard_id
                        )
    dataset = dataset.map(feature_fn,
                        input_columns=column_names,
                        output_columns=output_columns,
                        column_order=output_columns,
                        num_parallel_workers=num_workers
                        )
    type_cast_op = C.c_transforms.TypeCast(mstype.int32)
    for col in output_columns:
        dataset = dataset.map(operations=type_cast_op, input_columns=col)
    dataset = dataset.batch(batch_size=batch_size, drop_remainder=True)
    print_example(dataset)
    return dataset


def print_example(dataset, num_batch=1):
    iter = dataset.create_dict_iterator(output_numpy=True)
    for i in range(num_batch):
        print(next(iter))
    dataset.reset()
