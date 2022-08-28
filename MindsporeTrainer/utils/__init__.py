#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import os
from .logger_util import *
from .argument_types import *
from .xtqdm import *

def get_dir():
    cwd = os.path.dirname(os.path.abspath(__file__))
    cwd = cwd.split(cwd, os.path.sep)