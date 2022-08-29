# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# zbo@zju.edu.cn
# 2022-08-08
# ============================================================================

import os
from .logger_util import *
from .argument_types import *
from .xtqdm import *

def get_dir():
    cwd = os.path.dirname(os.path.abspath(__file__))
    cwd = cwd.split(cwd, os.path.sep)