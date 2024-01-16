# ---- coding: utf-8 ----
# @Author: Wu Yinpeng
# @Version: v01
# @Contact: YP_Wu@buaa.edu.cn
# @Date: 2022/9/6
"""
plot functions
"""
import sys
import numpy as np
import torch
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.animation as animation
from utils import path_join, get_project_root_path
# from data_process import de_pkg_y


def raw_data_visualization(dataset: str, index_list: list, fig_name_prefix: str, mode: str = '2d', show: bool = False):
    """
    plot raw data
    :param dataset: {'gaa', 'planar'}
    :param index_list: eg, [1, 3]
    :param fig_name_prefix: eg, 'exp1'
    :param mode: {'2d', '3d'}
        2d:
            plot id-vd, id-vg, respectively
        3d:
            x-y-z: vg-vd-id
    :param show: {True, False}
    """
    if mode == '2d':
        for index in index_list:
            txt_pth = path_join(get_project_root_path(), f"data/{dataset}/{index}.txt")
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 6))
            df = pd.read_table(txt_pth, sep='\t')

            # frozen `vg`, plot vd_ids (x-y)
            vg_set = set(df['vg'].values)
            for vg_val in vg_set:
                df_subset = df[df['vg'] == vg_val]
                ax1.plot(df_subset['vd'],
                         df_subset['ids'],
                         label=f"vg-{vg_val}")
            ax1.set_xlabel('Vd')
            ax1.set_ylabel('Ids')
            ax1.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
            plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

            ax1.set_title(f"Id-Vd")
            ax1.text(-0.1, 1.1, 'a', transform=ax1.transAxes, fontsize=16, fontweight='bold', va='top', ha='right')
            # ax1.title()
            ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))

            # froze `vd`, plot vg_ids (x-y)
            vd_set = set(df['vd'].values)
            for vd_val in vd_set:
                df_subset = df[df['vd'] == vd_val]
                ax2.plot(df_subset['vg'],
                         df_subset['ids'],
                         label=f"vd-{vd_val}")
            ax2.set_xlabel('Vg')
            ax2.set_ylabel('Ids')
            ax2.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
            plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))


            ax2.set_title(f"Id-Vg")
            ax2.text(-0.1, 1.1, 'b', transform=ax2.transAxes, fontsize=16, fontweight='bold', va='top', ha='right')
            # ax2.legend(loc='center left')
            ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))

            plt.subplots_adjust(wspace=0.3)
            plt.tight_layout()

            save_fig_pth = path_join(get_project_root_path(), f"figure/{fig_name_prefix}_{dataset}_2d_{index}.png")
            # save
            plt.gcf().savefig(save_fig_pth)
            if show:
                plt.show()
            plt.close('all')

    elif mode == '3d':
        for index in index_list:
            txt_pth = path_join(get_project_root_path(), f"data/{dataset}/{index}.txt")
            df = pd.read_table(txt_pth, sep='\t')

            # plot 3d vg-vd-ids(x-y-z)            
            ax = plt.axes(projection='3d')
            # ax.plot3D(df['vg'],
            #           df['vd'],
            #           df['ids'])
            
            vg, vd, ids = df['vg'].values, df['vd'].values, df['ids'].values
            
            n_differ_vd = len(np.unique(vd))
            
            vg = vg.reshape(-1, n_differ_vd)
            vd = vd.reshape(-1, n_differ_vd)
            ids = ids.reshape(-1, n_differ_vd)
            
            ax.plot_surface(vg, vd, ids, cmap='rainbow')
            
            ax.set_xlabel('Vg')
            ax.set_ylabel('Vd')
            ax.set_zlabel('Ids')
            ax.set_title('3D Visualization')

            save_fig_pth = path_join(get_project_root_path(), f"figure/{fig_name_prefix}dv_3d_{index}.png")
            # save
            plt.gcf().savefig(save_fig_pth)
            if show:
                plt.show()
            plt.close('all')


def plot_prediction_with_truth(
        dataset: str,
        plot_num: int,
        y_pred_info: tuple,
        fig_name_prefix: str,
        show: bool = True
):
    """
    plot 4 axis in one figure (2d), `frozen vg` & `frozen vd`, for prediction & truth
    to save disk, plot first `plot_num` figures (in index_list) only
    :param dataset: {'GAA', 'planar'}
    :param plot_num: first n figures, eg, 5 means plot first 5 figures in list
    :param y_pred_info: (y_pred, base), (torch.Tensor, str)
    :param fig_name_prefix: eg, 'exp2'
    :param show: bool, default, True
    """

    # read index.txt (in index_list), vg, vd, ids_truth
    para_txt_pth = path_join(get_project_root_path(), f"data/EDA2022/{dataset}_data/parametertest.txt")
    test_index_list = pd.read_table(para_txt_pth, sep=',')['index'].tolist()
    test_index_list = test_index_list[: plot_num]
    
    # pre-read n_rows of 1 index txt
    
    pre_df = pd.read_table(path_join(get_project_root_path(), f"data/EDA2022/{dataset}_data/1.txt"), sep='\t')
    n_rows = len(pre_df)  # 325
    
    y_pred = de_pkg_y(y_pred_info[0], y_pred_info[1])[: plot_num * n_rows, :]

    for i, abs_index in enumerate(test_index_list):
        index_txt_pth = path_join(get_project_root_path(), f"data/EDA2022/{dataset}_data/{abs_index}.txt")
        df = pd.read_table(index_txt_pth, sep='\t')
        fig, axs = plt.subplots(2, 2)

        def plot_frozen_v(v: str, ax, marker: str, y=None):
            """
            frozen vg or vd, for reproduction
            :param v: str, {'vg', 'vd'}
            :param ax: axes, {ax1, ax2, ax3, ax4}
            :param marker: {'truth', 'pred'}
            :param y: ease for plotting prediction
            """
            opposite_dict = {
                'vg': 'vd',
                'vd': 'vg'
            }
            # frozen v, plot opposite_dict['v']-ids (x-y)
            v_set = set(df[v].values)
            for v_val in v_set:
                df_subset = df[df[v] == v_val]
                if y is None:
                    ax.plot(df_subset[opposite_dict[v]],
                            df_subset['ids'],
                            label=f"{v}-{v_val}")
                elif isinstance(y, torch.Tensor):
                    # y, shape is like (25, 1)
                    df_copy = df.copy(deep=True)
                    df_copy['ids'] = pd.DataFrame(y)
                    df_copy_subset = df_copy[df_copy[v] == v_val]
                    ax.plot(df_subset[opposite_dict[v]],
                            df_copy_subset['ids'],
                            label=f"{v}-{v_val}")

                ax.set_xlabel(opposite_dict[v].capitalize())
                ax.set_ylabel('Ids')
                ax.set_title(f"{marker}-{opposite_dict[v].capitalize()}-Ids-{abs_index}")
                # ax.legend()

        # plot grid vg|vd-ids_truth
        plot_frozen_v('vg', axs[0, 0], 'truth')
        plot_frozen_v('vd', axs[0, 1], 'truth')
        # de pkg y_pred

        # plot grid vg|vd-ids_pred
        plot_frozen_v('vg', axs[1, 0], 'pred', y=y_pred[i*n_rows: (i+1)*n_rows, :])
        plot_frozen_v('vd', axs[1, 1], 'pred', y=y_pred[i*n_rows: (i+1)*n_rows, :])

        # setting
        plt.subplots_adjust(wspace=0.4, hspace=0.6)

        # save
        save_fig_pth = path_join(get_project_root_path(), f"figure/{fig_name_prefix}_absindex{abs_index}_truth_prediction.png")
        plt.gcf().savefig(save_fig_pth)

        # show, if needed
        if show:
            plt.show()

        plt.close('all')



def plot_tanh():
    x = np.linspace(-5, 5, 100)
    # 计算tanh函数的值
    y = np.tanh(x)

    # 绘制tanh函数的图像
    plt.figure(figsize=(8, 4))
    plt.plot(x, y, label="tanh(x)")
    plt.axhline(0, color='gray', lw=0.5)
    plt.axvline(0, color='gray', lw=0.5)
    plt.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
    plt.title("Tanh Function")
    plt.xlabel("x")
    plt.ylabel("tanh(x)")
    plt.legend()
    plt.show()

def plot_sigmoid():
    x = np.linspace(-5, 5, 100)
    # 计算tanh函数的值
    y = np.tanh(x)
    sigmoid = 1 / (1 + np.exp(-x))

    # 绘制sigmoid函数的图像
    plt.figure(figsize=(8, 4))
    plt.plot(x, sigmoid, label="sigmoid(x)")
    plt.axhline(0, color='gray', lw=0.5)
    plt.axhline(1, color='gray', lw=0.5, linestyle='--')
    plt.axvline(0, color='gray', lw=0.5)
    plt.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
    plt.title("Sigmoid Function")
    plt.xlabel("x")
    plt.ylabel("sigmoid(x)")
    plt.legend()
    plt.show()


def plot_cosine():
    def cosine_annealing_lr(t, lr_min, lr_max, T_max):
        return lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(t / T_max * np.pi))

    # 定义参数
    lr_min = 0.01
    lr_max = 0.1
    T_max = 100
    t = np.linspace(0, T_max, 1000)

    # 计算学习率
    lr = [cosine_annealing_lr(ti, lr_min, lr_max, T_max) for ti in t]

    # 绘制余弦退火学习率曲线
    plt.figure(figsize=(8, 4))
    plt.plot(t, lr)
    plt.title("Cosine Annealing Learning Rate")
    plt.xlabel("Iterations")
    plt.ylabel("Learning Rate")
    plt.grid(True)
    plt.show()


def plot_relu():

    def relu(x):
        return np.maximum(0, x)

    # 创建数据点
    x = np.linspace(-10, 10, 400)
    y = relu(x)
    
    # 绘制ReLU函数图像
    plt.figure(figsize=(8, 4))
    plt.plot(x, y, label="ReLU Function")
    plt.xlabel("x")
    plt.ylabel("ReLU(x)")
    plt.title("Graph of ReLU Function")
    plt.axhline(0, color='gray',linewidth=0.5)
    plt.axvline(0, color='gray',linewidth=0.5)
    plt.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
    plt.legend()
    plt.show()

def plot_logy():
    import os
    # planar
    y_test = torch.load("E:/PycharmProjects/GD_tcad_repo/data/planar/y_test.pt")
    # gaa
    # y_test = torch.load("E:/PycharmProjects/GD_tcad_repo/data/gaa/y_test.pt")
    x_index = y_test.shape[0]
    x = list(range(x_index))
    plt.plot(x, y_test, label="y_truth")

    # par_dir = "E:/PycharmProjects/GD_tcad_repo/results/cpu/planar/c5_epoch50"
    # model_names = os.listdir(par_dir)

    # for model in model_names:
    #     cur_pth = par_dir + f"/{model}/y_pred.pt"
    #     cur_logy_pred = torch.load(cur_pth)
    #     plt.plot(x, cur_logy_pred, label=f"{model}")

    plt.gcf().savefig("E:/PycharmProjects/GD_tcad_repo/figure/planar_logy.png")

    plt.show()
    plt.close('all')

class Animation():
    """
    Draw x and y to fig actively. You must init at least X and Y and , ax at first: fig, ax = plt.subplots()

    :param xlabel:
    :param ylabel:
    :param xlim:
    :param y:
    :return:
    """

    def __init__(self, xlabel=None, ylabel=None, xlim=None, ylim=None, xscale='linear',
                 yscale='linear', legend=None, figsize=None, X=[], Y=[]):
        if legend is None:
            legend = []
        self.fig, self.axes = plt.subplots(figsize=figsize)
        # 使用lambda函数捕获参数
        self.xlabel, self.ylabel = xlabel, ylabel
        self.xscale, self.yscale = xscale, yscale
        self.xlim, self.ylim = xlim, ylim
        self.legend = legend
        self.axes.grid()
        self.X, self.Y = X, Y
        self.xdata, self.ydata = [], []
        if len(self.Y) == 2:
            self.line = self.axes.plot([], [], 'b-', [], [], 'r-')
            y1, y2 = [], []
            self.ydata = [y1, y2]
        else:
            self.line, = self.axes.plot([], [], 'b-', lw=2)
        self.axes.grid()

    def draw(self, save=False):
        def init():
            self.axes.set_xlabel(self.xlabel)
            self.axes.set_ylabel(self.ylabel)
            self.axes.set_xscale(self.xscale)
            self.axes.set_yscale(self.yscale)
            self.axes.set_xlim(self.xlim)
            self.axes.set_ylim(self.ylim)
            # del self.xdata[:]
            # del self.ydata[:]
            if self.legend:
                self.axes.legend(self.legend)
            # return self.line,

        def update(num):
            # update the data
            t = self.X[num]
            xmin, xmax = self.axes.get_xlim()
            if t >= xmax:
                self.axes.set_xlim(xmin, 2 * xmax)
                self.axes.figure.canvas.draw()
            if len(self.Y) == 2:
                y1, y2 = self.Y[0][num], self.Y[1][num]
                self.xdata.append(t)
                self.ydata[0].append(y1)
                self.ydata[1].append(y2)
                self.line[0].set_data(self.xdata, self.ydata[0])
                self.line[1].set_data(self.xdata, self.ydata[1])
            else:
                y = self.Y[num]
                self.xdata.append(t)
                self.ydata.append(y)
                self.line.set_data(self.xdata, self.ydata)
            return self.line,

        ani = animation.FuncAnimation(self.fig, update, len(self.X), interval=100, init_func=init, repeat=False)
        if save is True:
            ani.save("test.gif", writer='imagemagick', fps=10)#保存
        plt.show(block=True)  # 显示


    
