# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# zbo@zju.edu.cn
# 2022-08-08
# ============================================================================

from mindspore.ops import operations as P
from mindspore.common import dtype as mstype

expand_dims = P.ExpandDims()
gather = P.Gather()
onehot = P.OneHot()
matmul = P.MatMul()
reshape = P.Reshape()
strided_slice = P.StridedSlice()
add = P.Add()
cast = P.Cast()
tile = P.Tile()
bml = P.BatchMatMul()
minimum = P.Minimum()
multiply = P.Mul()
transpose = P.Transpose()
get_dtype = P.DType()
alloc_status = P.NPUAllocFloatStatus()
get_status = P.NPUGetFloatStatus()
clear_status = P.NPUClearFloatStatus()
less_equal = P.LessEqual()
logical_or = P.LogicalOr()
not_equal = P.NotEqual()
select = P.Select()

_argmax = P.Argmax(axis=-1, output_type=mstype.int32)
def argmax(axis=-1, output_type=mstype.int32):
    return P.Argmax(axis=axis, output_type=output_type)
