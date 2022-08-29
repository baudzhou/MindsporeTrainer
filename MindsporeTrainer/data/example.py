# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# zbo@zju.edu.cn
# 2022-08-08
# ============================================================================

from collections import OrderedDict
import numpy as np
import numpy as np
import pickle

__all__=['ExampleInstance', 'example_to_feature', 'ExampleSet']

class ExampleInstance:
  def __init__(self, segments, label=None,  **kwv):
    self.segments = segments
    self.label = label
    self.__dict__.update(kwv)

  def __repr__(self):
    return f'segments: {self.segments}\nlabel: {self.label}'

  def __getitem__(self, i):
    return self.segments[i]

  def __len__(self):
    return len(self.segments)

class ExampleSet:
  def __init__(self, pairs):
    self._data = np.array([pickle.dumps(p) for p in pairs])
    self.total = len(self._data)

  def __getitem__(self, idx):
    """
    return pair
    """
    if isinstance(idx, tuple):
      idx,rng, ext_params = idx
    else:
      rng,ext_params=None, None
    content = self._data[idx]
    example = pickle.loads(content)
    segments = tuple([np.array(s) for s in example.segments])
    label = example.label
    if label is None:
      label = -1000
    return segments + (np.array([label]), )

  def __len__(self):
    return self.total

  def __iter__(self):
    for i in range(self.total):
      yield self[i]

def _truncate_segments(segments, max_num_tokens, rng):
  """
  Truncate sequence pair according to original BERT implementation:
  https://github.com/google-research/bert/blob/master/create_pretraining_data.py#L391
  """
  while True:
  #   if sum(len(s) for s in segments)<=max_num_tokens:
  #     break

    # segments = sorted(segments, key=lambda s:len(s), reverse=True)
    # trunc_tokens = segments[0]

    # assert len(trunc_tokens) >= 1
    if len(segments) <= max_num_tokens:
      break
    if rng.random() < 0.5:
      segments.pop(0)
    else:
      segments.pop()
  return segments

def truncate_tokens_pair(tokens_a, tokens_b, max_len, max_len_a=0, max_len_b=0, trunc_seg=None, always_truncate_tail=False):
    import random
    if isinstance(tokens_a, str) or isinstance(tokens_b, str):
      print('error')
    num_truncated_a = [0, 0]
    num_truncated_b = [0, 0]
    while True:
        if len(tokens_a) + len(tokens_b) <= max_len:
            break
        if (max_len_a > 0) and len(tokens_a) > max_len_a:
            trunc_tokens = tokens_a
            num_truncated = num_truncated_a
        elif (max_len_b > 0) and len(tokens_b) > max_len_b:
            trunc_tokens = tokens_b
            num_truncated = num_truncated_b
        elif trunc_seg:
            # truncate the specified segment
            if trunc_seg == 'a':
                trunc_tokens = tokens_a
                num_truncated = num_truncated_a
            else:
                trunc_tokens = tokens_b
                num_truncated = num_truncated_b
        else:
            # truncate the longer segment
            if len(tokens_a) > len(tokens_b):
                trunc_tokens = tokens_a
                num_truncated = num_truncated_a
            else:
                trunc_tokens = tokens_b
                num_truncated = num_truncated_b
        # whether always truncate source sequences
        if (not always_truncate_tail) and (random.random() < 0.5):
            del trunc_tokens[0]
            num_truncated[0] += 1
        else:
            trunc_tokens.pop()
            num_truncated[1] += 1
    return num_truncated_a, num_truncated_b

