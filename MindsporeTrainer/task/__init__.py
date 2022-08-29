# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# zbo@zju.edu.cn
# 2022-08-08
# ============================================================================

from .task import Task, TransformerTask, EvalData
from .task_registry import register_task, load_tasks, get_task
