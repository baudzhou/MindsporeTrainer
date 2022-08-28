
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# WYWEB Authors
# 
#

from glob import glob
import os
import importlib
import pdb
import sys
from loguru import logger
from .task import Task

__all__ = ['load_tasks', 'register_task', 'get_task']
tasks = {}

def register_task(name=None, desc=None):
  global tasks
  def register_task_x(cls):
    _name = name
    if _name is None:
      _name = cls.__name__

    _desc = desc
    if _desc is None:
      _desc = _name

    _name = _name.lower()
    if _name in tasks:
      logger.warning(f'{_name} already registered in the registry: {tasks[_name]}.')
    assert issubclass(cls, Task), f'Registered class must be a subclass of Task.'
    tasks[_name] = cls
    cls._meta = {
        'name': _name,
        'desc': _desc}
    return cls
  
  if type(name)==type:
    cls = name
    name = None
    return register_task_x(cls)
  return register_task_x

def load_tasks(task_dir=None, task_path=None):
  if task_path:
    if '.py' in task_path and os.path.exists(task_path):
      task_path = task_path.replace(os.getcwd(), '').replace('.py', '')
      m = task_path.replace(os.path.sep, '.')
      importlib.import_module(f'{m}')
      return
  script_dir = os.path.dirname(os.path.abspath(__file__))
  sys.path.append(script_dir)
  sys_tasks = glob(os.path.join(script_dir, "*.py"))
  for t in sys_tasks:
    m = os.path.splitext(os.path.basename(t))[0]
    if not m.startswith('_'):
      importlib.import_module(f'apps.tasks.{m}')

  if task_dir:
    assert os.path.exists(task_dir), f"{task_dir} must be a valid directory."
    customer_tasks = glob(os.path.join(task_dir, "*.py"))
    sys.path.append(task_dir)
    for t in customer_tasks:
      m = os.path.splitext(os.path.basename(t))[0]
      if not m.startswith('_'):
        importlib.import_module(f'{m}')

def get_task(name=None):
  global tasks
  if name is None:
    return tasks
  else:
    if '.py' in name and os.path.exists(name):
      return list(tasks.values())[0]
    else:
      return tasks[name.lower()]
