# ---- coding: utf-8 ----
# @Author: Wu Yinpeng
# @Version: v01
# @Contact: YP_Wu@buaa.edu.cn
# @Date: 2023/1/23

"""
functions for data pre-process
"""

import os
import torch
import pandas as pd
import numpy as np
from .utils import path_join, get_project_root_path, print_separate_line

torch.set_default_dtype(torch.double)
EPS = 1e-17


def id_reset0(
        df: pd.DataFrame
):
    """
    re-value ids to 0, when vd == 0
    :param df: DataFrame to be handled
    :return df_cleaned: DataFrame
    """
    df.loc[df['ids'] < 0, 'ids'] = EPS
    return df


# def get_params(
#         dataset: str,
#         mode: str
# ):
#     """
#     read 'parametertrain.txt' or 'parametertest.txt'
#     :param dataset: {'GAA', 'planar'}
#     :param mode: {'train', 'test'}
#     :return: [params(Tensor), index(list)]
#         params: (m, n)
#         index: (m, )
#     """
#     txt_pth = path_join(get_project_root_path(), f"data/{dataset}_data/parameter{mode}.txt")
#     df = pd.read_table(txt_pth, sep=',')
#     index_list = list(df.values[:, -1])
#     index_list = [int(item) for item in index_list]
#     return torch.Tensor(df.values[:, :-1]), index_list


def get_data(
        dataset: str,
        debug: bool = False
):
    """
    :param dataset: {'gaa', 'planar', 'circle', 'expand_circle', 'rectangle', 'triangle'}
    :param debug: bool

    :return: (X_train, X_test, y_train, y_test)
        X: [params(lg_nw, r_nw, etc.), vg, vd]
            (m, n)
        y: [ids]
            (m, dim_grid_vg_vd)

    """
    par_pth = path_join(get_project_root_path(), f"data/{dataset}")
    exist = False
    # check if data existed already, if does, read
    for root, _, files in os.walk(par_pth):
        if {'X_train.pt', 'X_test.pt', 'y_train.pt', 'y_test.pt'}.issubset(files):
            exist = True

    if exist and not debug:
        X_train = torch.load(path_join(par_pth, 'X_train.pt'))
        X_test = torch.load(path_join(par_pth, 'X_test.pt'))
        y_train = torch.load(path_join(par_pth, 'y_train.pt'))
        y_test = torch.load(path_join(par_pth, 'y_test.pt'))
        print("Data has been processed. Reading and delivering.")
        print_separate_line()
    else:
        # if not, pre-process data
        X_train, y_train = pkg_data(data_pth=path_join(par_pth, 'parametertrain.txt'))
        X_test, y_test = pkg_data(data_pth=path_join(par_pth, 'parametertest.txt'))
        torch.save(X_train, path_join(par_pth, 'X_train.pt'))
        torch.save(X_test, path_join(par_pth, 'X_test.pt'))
        torch.save(y_train, path_join(par_pth, 'y_train.pt'))
        torch.save(y_test, path_join(par_pth, 'y_test.pt'))
        print("Data all set. Saving and delivering.")
        print_separate_line()
    return X_train, X_test, y_train, y_test


def pkg_data(
        data_pth: str
):
    """
    pkg and return data, test frame version
    :param data_pth: eg, "E:\Dropbox\TCAD\data\EDA2022_primarius_v3\EDA2022\planar_data\parametertrain.txt"

    :return: (X, y)
        X: [params(vdd*, vdlin*, lg_nw, r_nw, etc.), vg, vd]
            (m, n)
        y: [ids]
            (m, dim_grid_vg_vd)
    """
    # 1. get index_list
    def get_params():
        df = pd.read_table(data_pth, sep=',')
        index_list = list(df.values[:, -1])
        index_list = [int(item) for item in index_list]
        return torch.Tensor(df.values[:, :-1]), index_list
    params, index_list = get_params()

    # define get par dir
    def get_par_dir():
        return os.path.abspath(os.path.join(data_pth, ".."))
    par_dir = get_par_dir()

    # 2. get_ids_with_index
    X, y, txt_samples = None, None, None
    exception_list = []
    for row, index in enumerate(index_list):
        item_pth = os.path.join(par_dir, f"{index}.txt")
        # item_pth = path_join(get_project_root_path(), f"data/{dataset}_data/{index}.txt")
        df = pd.read_table(item_pth, sep='\s+')
        df = id_reset0(df)
        # 3. pkg to tensor
        # 3.1 handle y
        item_y = torch.Tensor(df['ids']).reshape((-1, 1))
        item_x = torch.Tensor(df.values[:, :-1]).reshape((-1, 2))
        if row == 0:
            y = item_y
            X = item_x
            txt_samples = y.shape[0]
        else:
            if item_y.shape[0] != txt_samples:
                exception_list.append(row)
                print(f"Warning. Index {index} has {item_y.shape[0]} samples({txt_samples} expected), "
                      f"non-convergence happens.")
                continue
            else:
                y = torch.cat((y, item_y), dim=0)
                X = torch.cat((X, item_x), dim=0)

    print(f"{len(exception_list)} exceptions happened.")

    # 3.3 concat parameters_basic, handle exception if list has items
    if exception_list:
        # delete params(re-read parameters), according to exception_list
        # txt_pth = path_join(get_project_root_path(), f"data/{dataset}_data/parameter{tr_te_mode}.txt")
        df = pd.read_table(data_pth, sep=',')
        df = df.drop(df.index[exception_list])
        params = torch.Tensor(df.values[:, :-1])

    params_expand = None
    p_dims = params.shape[1]
    for p_row in range(params.shape[0]):
        # expand whole rows
        if p_row == 0:
            params_expand = params[p_row, :].expand((txt_samples, p_dims))
        else:
            params_expand = torch.cat((params_expand, params[p_row, :].expand((txt_samples, p_dims))), dim=0)
    params = params_expand
    X = torch.cat((params, X), dim=1)

    # 3.5 transform data if needed
    method_dict = {'2_log': torch.log2,
                   '2_exp': torch.exp2,
                   'e_log': torch.log,
                   'e_exp': torch.exp}
    # handle X
    df_op = pd.DataFrame(X)
    # 3.5.1 if X > 1e2, log
    condition_list = []
    for col in df_op.columns.tolist():
        if df_op.values[0, col] > 1e2:
            condition_list.append(col)
    for col_index in condition_list:
        df_op[col_index] = method_dict[f"e_log"](torch.Tensor(df_op[col_index]))

    # 3.5.2 others, exp (planar only)
    if 'gaa' not in data_pth:
        for col in df_op.columns.tolist():
            if col not in condition_list:
                df_op[col] = method_dict[f"e_exp"](torch.Tensor(df_op[col]))

    X = torch.Tensor(df_op.values)

    # 3.5.3 handle y, log
    y = method_dict[f"e_log"](y)

    return X, y


def de_pkg_y(data: torch.Tensor, base: str):
    """
    restore y (exp)
    :param data: y (prediction), (m, n)
    :param base: {'2', 'e'}
    :return: y_ids
    """
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)
    method = torch.exp if base == 'e' else torch.exp2
    return method(data)


def split_vd0_item(data: torch.Tensor, dataset: str):
    """
    split data into `vd==0 data` & `left data`
    :param data: X or y
    :param dataset: {'gaa', 'planar'}
    :return: vd0_data, left_data
    """
    # get vd nums
    any_txt = path_join(get_project_root_path(), f"./data/{dataset}/1.txt")
    vd_set_nums = len(set(pd.read_table(any_txt, sep='\t')['vd']))
    df = pd.DataFrame(data)
    vd0 = df.iloc[df.index % vd_set_nums == 0]
    drop_list = []
    for row in range(data.shape[0]):
        if row % vd_set_nums == 0:
            drop_list.append(row)
    left = df.drop(df.index[drop_list])
    return torch.Tensor(vd0.values), torch.Tensor(left.values)


def make_up_vd0_item(vd0: torch.Tensor, left: torch.Tensor, txt_line: int):
    """
    make up regular data with vd0 data
    NOTICE < this function only serve for cases where `remove_ids0` called
    :param vd0: vd0 data
    :param left: regular data
    :param txt_line: eg, 325
    :return: whole data
    """
    vd_counter, left_counter = 0, 0
    vd_max_samples, left_max_samples = vd0.shape[0], left.shape[0]
    whole_data = torch.ones((vd0.shape[0] + left.shape[0], 1))

    for row_index in range(whole_data.shape[0]):
        if row_index % txt_line == 0:
            if vd_counter < vd_max_samples:
                whole_data[row_index] = vd0[vd_counter]
                vd_counter += 1
            else:
                continue
        else:
            if left_counter < left_max_samples:
                whole_data[row_index] = left[left_counter]
                left_counter += 1
    return whole_data


def remove_ids0(X: torch.Tensor, y: torch.Tensor, base: str):
    """
    remove data which ids is zero, portable for `overlay` mode
    return left data
    :param X: X, (n, X_dim)
    :param y: y, (n, 1)
    :param base: {'2', 'e'}

    :return X_left, y_left
    """
    index_list = []
    method_dict = {'2_log': torch.log2,
                   'e_log': torch.log}
    for row in range(y.shape[0]):
        if y[row].item() == method_dict[f"{base}_log"](torch.Tensor([EPS])).item():
            index_list.append(row)
    df_X = pd.DataFrame(X)
    df_y = pd.DataFrame(y)
    X_left = torch.Tensor(df_X.drop(index_list).values)
    y_left = torch.Tensor(df_y.drop(index_list).values)
    return X_left, y_left







