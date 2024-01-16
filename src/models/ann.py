# ---- coding: utf-8 ----
# @Author: Wu Yinpeng
# @Version: v01
# @Contact: YP_Wu@buaa.edu.cn
# @Date: 2023/1/24

"""
Artificial Neural Network Definition
train & validation watch
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
# from models.model_base import ModelBase
from model_base import ModelBase


class ANN(nn.Module):
    def __init__(self, X_tr, y_tr, X_te, y_te, batch_size, mode=0, norm=True):
        super(ANN, self).__init__()
        self.name = 'ann'
        self.norm = norm
        if norm:
            # normalization (by each dim)
            X_tr_norm = (X_tr - X_tr.mean(0).expand_as(X_tr)) / (X_tr.std(0).expand_as(X_tr))
            X_te_norm = (X_te - X_te.mean(0).expand_as(X_te)) / (X_te.std(0).expand_as(X_te))

            self.y_tr_mean = y_tr.mean(0)  # scalar
            self.y_tr_std = y_tr.std(0)  # scalar
            y_tr_norm = (y_tr - self.y_tr_mean.expand_as(y_tr)) / self.y_tr_std.expand_as(y_tr)

            y_te_norm = (y_te - y_te.mean(0).expand_as(y_te)) / (y_te.std(0).expand_as(y_te))

            self.train_loader = DataLoader(TensorDataset(X_tr_norm, y_tr_norm), batch_size, shuffle=True)
            self.test_loader = DataLoader(TensorDataset(X_te_norm, y_te_norm), batch_size, shuffle=True)
        else:
            # no normalization
            self.train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size, shuffle=True)
            self.test_loader = DataLoader(TensorDataset(X_te, y_te), batch_size, shuffle=True)

        self.X_dim = X_tr.shape[1]  # 8 or 5
        self.y_dim = y_tr.shape[1]  # 1
        self.batch_size = batch_size
        self.hidden_layer_dim = X_tr.shape[1] * 2
        if mode == 0:  # define 0
            self.work_flow = nn.Sequential(
                # hidden layer1
                nn.Linear(self.X_dim, self.hidden_layer_dim),
                nn.LogSigmoid(),
                # hidden layer2
                nn.Linear(self.hidden_layer_dim, self.X_dim),
                nn.LogSigmoid(),
                # output layer
                nn.Linear(self.X_dim, self.y_dim)
            )
        elif mode == 1:
            self.work_flow = nn.Sequential(
                # hidden layer1
                nn.Linear(self.X_dim, self.hidden_layer_dim),
                nn.ReLU(),
                # hidden layer2
                nn.Linear(self.hidden_layer_dim, self.hidden_layer_dim),
                nn.ReLU(),
                # hidden layer3
                nn.Linear(self.hidden_layer_dim, self.X_dim),
                nn.ReLU(),
                # output layer
                nn.Linear(self.X_dim, self.y_dim)
            )
        elif mode == 2:
            self.work_flow = nn.Sequential(
                # hidden layer1
                nn.Linear(self.X_dim, self.X_dim),
                nn.ReLU(),
                # hidden layer2
                nn.Linear(self.X_dim, self.X_dim),
                nn.ReLU(),
                # output layer
                nn.Linear(self.X_dim, self.y_dim)
            )

    def forward(self, x):
        return self.work_flow(x)
#
#     def fit(self, lr, epoch, opt: str = 'sgd'):
#         # train ANN
#         self.train()
#         optimizer = None
#         if opt == 'sgd':
#             optimizer = torch.optim.SGD(self.parameters(), lr=lr)
#         elif opt == 'adam':
#             optimizer = torch.optim.Adam(self.parameters(), lr=lr)
#         loss_fnc = nn.MSELoss()
#
#         for e in range(epoch):
#             print(f"Epoch {e + 1}\n----------------------------------------------")
#             size = len(self.train_loader.dataset)
#             for batch, (X, y) in enumerate(self.train_loader):
#                 optimizer.zero_grad()
#                 # compute loss, BP
#                 y_pred = self(X)
#                 # for name, para in self.named_parameters():
#                 #     print(name, para)
#                 loss = loss_fnc(y_pred, y) / y.shape[0]
#                 loss.backward()
#                 optimizer.step()
#                 if batch % 50 == 0:
#                     loss, cur = loss.item(), batch * len(X)
#                     print(f"Loss: {loss:>5f}\t\t [{cur:>5d} / {size:>5d}]")
#             self.test()
#
#     def test(self):
#         self.eval()
#         test_loss = 0
#         loss_fnc = nn.MSELoss()
#         with torch.no_grad():
#             for X, y in self.test_loader:
#                 y_pred = self(X)
#                 # print(y_pred)
#                 test_loss += loss_fnc(y_pred, y).item() / y.shape[0]
#         test_loss /= len(self.test_loader)
#         print(f"Test MSE Loss: {test_loss:>8f}")
#
#     def predict(self, X_te):
#         with torch.no_grad():
#             if self.norm:
#                 X_te_norm = (X_te - X_te.mean(0).expand_as(X_te)) / (X_te.std(0).expand_as(X_te))
#                 y_pred_revert = self(X_te_norm) * \
#                                 self.y_tr_std.expand((X_te.shape[0], self.y_dim)) + \
#                                 self.y_tr_mean.expand((X_te.shape[0], self.y_dim))
#             else:
#                 y_pred_revert = self(X_te)
#             return y_pred_revert


class ANNX(ModelBase):

    def __init__(self, input_dim, output_dim):
        
        super(ANNX, self).__init__()

        self.name = 'ann'
        self.loss_func = self.customed_loss()
        
        self.X_dim = input_dim  # 8 or 5
        self.y_dim = output_dim  # 1

        self.hidden_layer_dim = input_dim * 2
        
        # self.hidden_layer_dim = 4

        # TODO: self-adaptive adjust net structure0
        self.work_flow = nn.Sequential(
            # input layer
            nn.Linear(self.X_dim, self.hidden_layer_dim),
            nn.BatchNorm1d(self.hidden_layer_dim),
            # nn.ReLU(),
            nn.Sigmoid(),

            # hidden layer1
            nn.Linear(self.hidden_layer_dim, self.hidden_layer_dim),
            nn.BatchNorm1d(self.hidden_layer_dim),
            # nn.ReLU(),
            nn.Sigmoid(),

            # # # hidden layer2
            nn.Linear(self.hidden_layer_dim, self.hidden_layer_dim),
            nn.BatchNorm1d(self.hidden_layer_dim),
            # nn.ReLU(),
            nn.Sigmoid(),

            nn.Linear(self.hidden_layer_dim, self.hidden_layer_dim),
            nn.BatchNorm1d(self.hidden_layer_dim),
            # nn.ReLU(),
            nn.Sigmoid(),

            # output layer
            nn.Linear(self.hidden_layer_dim, self.y_dim)
        )

    def forward(self, x):
        return self.work_flow(x)

    def customed_loss(self):
        return torch.nn.MSELoss()


if __name__ == '__main__':

    torch.manual_seed(1)
    x = torch.randn((5, 8))
    print(x)
    model = ANNX(8, 1)
    out = model(x)
    torch.save(model, 'ann.pt')
    print(f"out is {out}")

