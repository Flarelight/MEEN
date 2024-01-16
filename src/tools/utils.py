# ---- coding: utf-8 ----
# @Author: Wu Yinpeng
# @Version: v01
# @Contact: YP_Wu@buaa.edu.cn
# @Date: 2023/1/23

"""
intro of this file
"""

import os
import re
import time
import torch
import numpy as np
from pathlib import Path
import platform
from sklearn.metrics import r2_score, mean_squared_error


def print_separate_line():
    line = ''
    for i in range(20):
        line += '-----'
    print(line)


def print_metrics(marker: str, y_truth: np.ndarray, y_pred: np.ndarray):
    r2 = r2_score(y_truth, y_pred)
    mse = mean_squared_error(y_truth, y_pred)
    print(f"For model {marker}\nR2 = {r2}\nMSE = {mse}")
    return r2, mse


def get_project_root_path(project_name: str = 'GD_tcad_repo') -> str:
    # project_name: 'STA_Multitask'
    cur_path = os.path.abspath(os.path.dirname(__file__))
    os_platform = platform.system().lower()
    separator = '\\' if os_platform == 'windows' else '/'
    pattern = f".*{project_name}\\{separator}" if os_platform == 'windows' else f".*{project_name}{separator}"
    root_path = None
    try:
        root_path = re.match(pattern, cur_path).group()
    except AttributeError:
        print("Argument `project_name` not exists. Please check.")
    return root_path


def get_time_stamp() -> str:
    return time.strftime("%Y%m%d%H%M%S")


def fix_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def path_join(pth1, pth2):
    return Path(os.path.join(pth1, pth2)).as_posix()

