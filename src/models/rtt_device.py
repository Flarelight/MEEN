# ---- coding: utf-8 ----
# @Author: Wu Yinpeng
# @Version: v01
# @Contact: YP_Wu@buaa.edu.cn
# @Date: 2023/6/10

"""
intro of this file
"""

import torch
import torch.nn as nn
import math
from models.model_base import ModelBase


class PreprocessorWithConv1d(torch.nn.Module):
    
    def __init__(self, x_dim):
        
        self.ln = torch.nn.LayerNorm(x_dim)
        sqrt_input_dim = math.sqrt(x_dim)
        self.conv = torch.nn.Conv1d(in_channels=sqrt_input_dim * 2,
                                    out_channels=sqrt_input_dim,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1)
    
    def forward(self, x):
        # ln, (n, d) -> (n, d, 1)
        ln_output = self.ln(x)
        # conv, (n, d, 1) -> (n, c_d, 1)
        return self.conv(ln_output.unsqueeze(-1)).squeeze(-1)


class RTTDeviceModel(ModelBase):
    """_summary_
    RTT-device model
    LN + (NN-extractor + CNN) + RNN + derivative loss
    """
    def __init__(self,
                 input_dim,
                 hidden_dim: int = 8,
                 num_layers: int = 4,
                 activation : str = 'tanh',
                 dropout: float = 0,
                 conv: bool = False):
        
        assert activation in ['tanh', 'relu']
        super(RTTDeviceModel, self).__init__()
        
        self.name = 'rtt_device'
        self.train_loss = self.customed_loss()
        self.eval_loss = nn.MSELoss()
        self.preprocessor, self.predictor = None, None
        
        if conv:
            self.preprocessor = PreprocessorWithConv1d(input_dim)
            self.predictor = torch.nn.RNN(input_size=math.sqrt(input_dim),
                                          hidden_size=hidden_dim,
                                          num_layers=num_layers,
                                          nonlinearity=activation,
                                          dropout=dropout)
        else:
            self.preprocessor = torch.nn.LayerNorm(input_dim)
            self.predictor = torch.nn.RNN(input_size=input_dim,
                                          hidden_size=hidden_dim,
                                          num_layers=num_layers,
                                          nonlinearity=activation,
                                          dropout=dropout)
        self.collector = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, h):
        # preprocessor, (n, d1) -> (n_d2)
        # predictor, (n, d2) -> (n, r_d)
        # collector, (n, r_d) -> (n, 1)
        
        prep_out = self.preprocessor(x)
        pred_out, h = self.predictor(prep_out, h)
        y_pred = self.collector(pred_out)
        
        return y_pred, h


    def customed_loss(self): # TODO: 2023-06-13 19:20:22
        # opt1, derative loss
        # return RTTDeviceDerivLoss()
        
        # opt2, nll
        return torch.nn.MSELoss(reduction='sum')
    