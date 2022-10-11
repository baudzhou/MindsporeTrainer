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
from mindspore.dataset.vision.c_transforms import ImageTensorOperation
import mindspore.common.dtype as mstype
logger=loguru.logger

def create_dynamic_dataset(examples, feature_fn, batch_size, output_columns, column_names=["example", "label"],
                           buffer_size=1000, repeat=1, num_workers=1, num_shards=None, python_multiprocessing=True,
                           shard_id=None, type_cast_op=[C.c_transforms.TypeCast(mstype.int32)], shuffle=True):
    '''
        Provide dynamic data set.
        Args:
            examples: text example list
            feature_fn: function for generating mask etc.
    '''
    dataset = ds.GeneratorDataset(examples, 
                                  shuffle=shuffle,
                                  column_names=column_names,
                                  num_shards=num_shards,
                                  shard_id=shard_id,
                                  num_parallel_workers=num_workers
                                )
    # test_feature_fn(examples, feature_fn)
    for fn, input_column, output_column in feature_fn:
        if isinstance(fn, ImageTensorOperation):
            output_column = None
        dataset = dataset.map(fn,
                              input_columns=input_column,
                              output_columns=output_column,
                              column_order=output_column,
                              num_parallel_workers=num_workers,
                              python_multiprocessing=python_multiprocessing
                            )
    # print_example(dataset)
    if type_cast_op is not None:
        for op, col in type_cast_op:
            dataset = dataset.map(operations=op, input_columns=col, num_parallel_workers=num_workers)
    dataset = dataset.batch(batch_size=batch_size, drop_remainder=True, 
                            num_parallel_workers=num_workers, output_columns=output_columns,
                            column_order=output_columns)
    # print_example(dataset)
    return dataset


def print_example(dataset, num_batch=1):
    iter = dataset.create_dict_iterator(output_numpy=True)
    for i in range(num_batch):
        e = next(iter)
        print(f'sample shape: {e["img"].shape}\nlabels: {e["label"]}')
    dataset.reset()

def test_feature_fn(example, feature_fn):
    e = example[0]
    for fn, name in feature_fn:
        e = fn(*e)
    print(e)