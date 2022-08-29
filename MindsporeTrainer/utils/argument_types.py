# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# zbo@zju.edu.cn
# 2022-08-08
# ============================================================================

def boolean_string(s):
  if s.lower() not in {'false', 'true'}:
    raise ValueError('Not a valid boolean string')
  return s.lower() == 'true'
