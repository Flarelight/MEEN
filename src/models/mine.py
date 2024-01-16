
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .model_base import ModelBase
from .cigp_v14 import CIGP


class SelfAttention(nn.Module):
    dim_in: int
    dim_k: int
    dim_v: int

    def __init__(self, dim_in, dim_k, dim_v):
        super(SelfAttention, self).__init__()
        self.dim_in = dim_in
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.linear_q = nn.Linear(dim_in, dim_k, bias=False)
        self.linear_k = nn.Linear(dim_in, dim_k, bias=False)
        self.linear_v = nn.Linear(dim_in, dim_v, bias=False)
        self._norm_fact = 1 / math.sqrt(dim_k)

    def forward(self, x):
        # x: batch, n, dim_in
        batch, n, dim_in = x.shape
        assert dim_in == self.dim_in

        q = self.linear_q(x)  # batch, n, dim_k
        k = self.linear_k(x)  # batch, n, dim_k
        v = self.linear_v(x)  # batch, n, dim_v

        dist = torch.bmm(q, k.transpose(1, 2)) * self._norm_fact  # batch, n, n
        dist = torch.softmax(dist, dim=-1)  # batch, n, n

        att = torch.bmm(dist, v)
        return att


class MyModel1(ModelBase):
    """
    2-tower mixture experts
    opt1, base: (2 h)
    y = attention([RNN(para, vd) * vg, RNN(para, vg) * vd, ANN(para, vg, vd)]) -> ReLU -> Linear
    
    opt2, residual: (3 h)
    y = attention([RNN(para, vd) * vg, RNN(para, vg) * vd, ANN(para, vg, vd)]) -> RelU -> Linear + RNN(para, vg, vd)
    
    opt3, moe: (3 h)
    y = w1* RNN(para, vd) + w2* RNN(para, vg) + w3* RNN(para, vg, vd)
    """
    def __init__(self, 
                 input_dim, 
                 hidden_dim: int = 8, 
                 num_layers: int = 4, 
                 dropout: float = 0, 
                 mode: str = 'base'):
        
        super(MyModel1, self).__init__()
        
        assert mode in ['base', 'res', 'moe']
        
        self.name = 'mine'
        self.loss_func = self.customed_loss()
        
        self.mode = mode
        
        self.ln = torch.nn.LayerNorm(input_dim)
        self.attention_module = SelfAttention(dim_in=3*hidden_dim, dim_k=hidden_dim, dim_v=hidden_dim)
        self.vg_tower = torch.nn.GRU(input_size=input_dim-1,
                                     hidden_size=hidden_dim,
                                     num_layers=num_layers,
                                     dropout=dropout)
        self.vd_tower = torch.nn.GRU(input_size=input_dim-1,
                                     hidden_size=hidden_dim,
                                     num_layers=num_layers,
                                     dropout=dropout)
        self.center_tower = torch.nn.Sequential(
            # input layer
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),

            # hidden layer1
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),

            # hidden layer2
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            
            # output layer
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.output_layer = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1)
        )
        
        if mode == 'base':
            self.aux_tower = None
        elif mode == 'res':
            self.aux_tower = torch.nn.GRU(input_size=input_dim,
                                          hidden_size=hidden_dim,
                                          num_layers=num_layers,
                                          dropout=dropout)
        elif mode == 'moe':
            self.aux_tower = torch.nn.GRU(input_size=input_dim,
                                          hidden_size=hidden_dim,
                                          num_layers=num_layers,
                                          dropout=dropout)
            self.w1 = torch.nn.Parameter(torch.ones(1, 1))
            self.w2 = torch.nn.Parameter(torch.ones(1, 1))
            self.w3 = torch.nn.Parameter(torch.ones(1, 1))


    def forward(self, x, vg_t_h = None, vd_t_h = None, aux_t_h = None):
        # LN
        ln_out = self.ln(x)
        # assuming x is like [para_1, ..., para_n, vg, vd]
        # construct [para_1, ..., para_n, vg] & [para_1, ..., para_n, vd]
        para_vg = ln_out[:, :-1]
        para_vd = torch.concat((ln_out[:, :-2], ln_out[:, -1].unsqueeze(-1)), -1)
        vg = ln_out[:, -2].unsqueeze(-1)
        vd = ln_out[:, -1].unsqueeze(-1)
        
        # vg tower, (n, hidden)
        vg_t_out, vg_t_h = self.vg_tower(para_vd, vg_t_h)
        
        # vd tower, (n, hidden)
        vd_t_out, vd_t_h = self.vd_tower(para_vg, vd_t_h)
        
        # ann, (n, hidden)
        center_out = self.center_tower(ln_out)
        
        if self.mode == 'base':
            # (n, hidden*3) -> (n, 1, hidden*3)
            attention_input = torch.concat([vg_t_out * vd, vd_t_out * vg, center_out], -1).unsqueeze(1)
            # (n, 1, hidden) -> (n, hidden)
            attention_output = self.attention_module(attention_input).squeeze(1)
            y_pred = self.output_layer(attention_output)
            
            return y_pred, vg_t_h, vd_t_h

        elif self.mode == 'res':
            # (n, hidden*3) -> (n, 1, hidden*3)
            attention_input = torch.concat([vg_t_out * vd, vd_t_out * vg, center_out], -1).unsqueeze(1)
            # (n, 1, hidden) -> (n, hidden)
            attention_output = self.attention_module(attention_input).squeeze(1)
            
            # RNN, (n, hidden)
            aux_out, aux_t_h = self.aux_tower(ln_out, aux_t_h)
            # (n, hidden)
            y_pred = self.output_layer(attention_output) + aux_out
            y_pred = self.output_layer(y_pred)
            
            return y_pred, vg_t_h, vd_t_h, aux_t_h

        elif self.mode == 'moe':
            # TODO: this is not mixture-of-experts !
            # (n, hidden) + (n, hidden) + (n, hidden)
            
            # RNN, (n, hidden)
            aux_out, aux_t_h = self.aux_tower(ln_out)
            # (n, hidden)
            y_pred = self.w1 * vg_t_out + self.w2 * vd_t_out + self.w3 * aux_out
            y_pred = self.output_layer(y_pred)
            
            return y_pred, vg_t_h, vd_t_h, aux_t_h



    def customed_loss(self):
        
        # opt1, mse
        return torch.nn.MSELoss()
        
        # opt2, derivative loss
        
        pass
        





class MyModel2(ModelBase):
    
    """
    arch:

    NN[NN(vg, vd), NN[para, AE(para)]]

    output(gd_NN(vg, vd), para_NN(para, encoderNN(para)))
    """
    def __init__(self, input_dim):
        super(MyModel2, self).__init__()
        
        self.name = 'mine2_AE'
        # self.train_loss = self.customed_loss()
        # self.eval_loss = nn.MSELoss()
        
        self.input_dim = input_dim
        self.gd_dim = 4
        self.hidden_dim = input_dim * 2
        self.para_dim = input_dim - 2
        # self.encode_dim = int(math.sqrt(self.para_dim))

        self.encode_dim = 4

        self.loss_lambda = nn.Parameter(torch.tensor([0.01]))
        
        self.gd_nn = nn.Sequential(
            # input
            nn.Linear(2, self.gd_dim),
            nn.BatchNorm1d(self.gd_dim),
            # nn.LogSigmoid(),
            nn.ReLU(),
            
            # hidden layer1
            nn.Linear(self.gd_dim, self.gd_dim),
            nn.BatchNorm1d(self.gd_dim),
            # nn.LogSigmoid(),
            nn.ReLU(),

            # hidden layer2
            nn.Linear(self.gd_dim, self.gd_dim),
            nn.BatchNorm1d(self.gd_dim),
            # nn.LogSigmoid(),
            nn.ReLU(),
            
            # hidden layer3
            nn.Linear(self.gd_dim, self.gd_dim),
            nn.BatchNorm1d(self.gd_dim),
            # nn.LogSigmoid(),
            nn.ReLU(),
            
            nn.Linear(self.gd_dim, self.gd_dim)
        )
        
        self.para_encoder = nn.Sequential(
            nn.Linear(self.para_dim, self.encode_dim*2),
            nn.ReLU(),
            nn.Linear(self.encode_dim*2, self.encode_dim)
        )
        self.para_decoder = nn.Sequential(
            nn.Linear(self.encode_dim, self.encode_dim*2),
            nn.ReLU(),
            nn.Linear(self.encode_dim*2, self.para_dim)
        )
        
        self.parann_input_dim = self.encode_dim + self.para_dim
        
        self.para_nn = nn.Sequential(
            # input
            nn.Linear(self.parann_input_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            
            # hidden layer1
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),

            # hidden layer2
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),

            nn.Linear(self.hidden_dim, self.input_dim)
        )
        
        outlayer_indim = self.gd_dim + self.input_dim
        self.output_layer = nn.Sequential(
            nn.BatchNorm1d(self.gd_dim + self.input_dim),
            nn.ReLU(),
            nn.Linear(outlayer_indim, 1)
        )


    def forward(self, x):
        para = x[:, :-2]
        vgvd = x[:, -2:]
        
        # gdnn
        gdnn_out = self.gd_nn(vgvd)
        
        # para_encoder
        para_code = self.para_encoder(para)
        
        # parann
        parann_out = self.para_nn(torch.cat((para, para_code), -1))
        
        # output layer
        y_pred = self.output_layer(torch.cat((gdnn_out, parann_out), -1))
        
        return y_pred
    
    
    def customed_loss(self):

        class my_loss(nn.Module):
            def __init__(self, auto_encoder):
                super().__init__()
                self.encoder = auto_encoder[0]
                self.decoder = auto_encoder[1]

            def forward(self, x, y_pred, y_truth):
                x_para = x[:, :-2]
                x_vgvd = x[:, -2:]

                # truth loss
                loss_truth = nn.MSELoss()
                # derivative loss，前提条件：需要x在反向传播的时候
                loss_deri = nn.MSELoss()
                
                # 对两个loss求和
                return loss_truth(y_pred, y_truth) + loss_deri(y_pred, y_truth)
                # return torch.sqrt(loss_ae(decoder_output, x_para)**2 + loss_deri(y_pred, y_truth)**2)

        return my_loss([self.para_encoder, self.para_decoder])


class MyModel3(ModelBase):
    
    """
    arch:

    output(NN[encoder(para), vg, vd])

    """
    def __init__(self, input_dim):
        super(MyModel3, self).__init__()
        
        self.name = 'mine3_AE'
        self.loss_func = self.customed_loss()
        
        self.input_dim = input_dim
        self.gd_dim = 4
        self.hidden_dim = input_dim * 2
        self.para_dim = input_dim - 2
        self.encode_dim = int(math.sqrt(self.para_dim))
        
        self.gd_nn = nn.Sequential(
            # input
            nn.Linear(2, self.gd_dim),
            nn.BatchNorm1d(self.gd_dim),
            nn.ReLU(),
            
            # hidden layer1
            nn.Linear(self.gd_dim, self.gd_dim),
            nn.BatchNorm1d(self.gd_dim),
            nn.ReLU(),

            # hidden layer2
            nn.Linear(self.gd_dim, self.gd_dim),
            nn.BatchNorm1d(self.gd_dim),
            nn.ReLU(),
            
            # hidden layer3
            nn.Linear(self.gd_dim, self.gd_dim),
            nn.BatchNorm1d(self.gd_dim),
            nn.ReLU(),
            
            nn.Linear(self.gd_dim, self.gd_dim)
        )
        
        self.para_encoder = nn.Sequential(
            nn.Linear(self.para_dim, self.encode_dim*2),
            nn.ReLU(),
            nn.Linear(self.encode_dim*2, self.encode_dim)
        )
        self.para_decoder = nn.Sequential(
            nn.Linear(self.encode_dim, self.encode_dim*2),
            nn.ReLU(),
            nn.Linear(self.encode_dim*2, self.para_dim)
        )
        
        self.parann_input_dim = self.encode_dim + self.para_dim
        
        self.para_nn = nn.Sequential(
            # input
            nn.Linear(self.parann_input_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            
            # hidden layer1
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),

            # hidden layer2
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),

            nn.Linear(self.hidden_dim, self.input_dim)
        )
        
        outlayer_indim = self.gd_dim + self.input_dim
        self.output_layer = nn.Sequential(
            nn.BatchNorm1d(self.gd_dim + self.input_dim),
            nn.ReLU(),
            nn.Linear(outlayer_indim, 1)
        )


    def forward(self, x):
        para = x[:, :-2]
        vgvd = x[:, -2:]
        
        # gdnn
        gdnn_out = self.gd_nn(vgvd)
        
        # para_encoder
        para_code = self.para_encoder(para)
        
        # parann
        parann_out = self.para_nn(torch.cat((para, para_code), -1))
        
        # output layer
        y_pred = self.output_layer(torch.cat((gdnn_out, parann_out), -1))
        
        return y_pred


class MyModel4(ModelBase):
    
    """
    arch:

    output(LSTM[encoder(para), vg, vd])

    """
    def __init__(self, input_dim):
        super(MyModel4, self).__init__()
        
        self.name = 'mine3'
        self.loss_func = self.customed_loss()
        
        self.input_dim = input_dim
        self.gd_dim = 4
        self.hidden_dim = input_dim * 2
        self.para_dim = input_dim - 2
        self.encode_dim = int(math.sqrt(self.para_dim))
        
        self.gd_nn = nn.Sequential(
            # input
            nn.Linear(2, self.gd_dim),
            nn.BatchNorm1d(self.gd_dim),
            nn.ReLU(),
            
            # hidden layer1
            nn.Linear(self.gd_dim, self.gd_dim),
            nn.BatchNorm1d(self.gd_dim),
            nn.ReLU(),

            # hidden layer2
            nn.Linear(self.gd_dim, self.gd_dim),
            nn.BatchNorm1d(self.gd_dim),
            nn.ReLU(),
            
            # hidden layer3
            nn.Linear(self.gd_dim, self.gd_dim),
            nn.BatchNorm1d(self.gd_dim),
            nn.ReLU(),
            
            nn.Linear(self.gd_dim, self.gd_dim)
        )
        
        self.para_encoder = nn.Sequential(
            nn.Linear(self.para_dim, self.encode_dim*2),
            nn.ReLU(),
            nn.Linear(self.encode_dim*2, self.encode_dim)
        )
        self.para_decoder = nn.Sequential(
            nn.Linear(self.encode_dim, self.encode_dim*2),
            nn.ReLU(),
            nn.Linear(self.encode_dim*2, self.para_dim)
        )
        
        self.parann_input_dim = self.encode_dim + self.para_dim
        
        self.para_nn = nn.Sequential(
            # input
            nn.Linear(self.parann_input_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            
            # hidden layer1
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),

            # hidden layer2
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),

            nn.Linear(self.hidden_dim, self.input_dim)
        )
        
        outlayer_indim = self.gd_dim + self.input_dim
        self.output_layer = nn.Sequential(
            nn.BatchNorm1d(self.gd_dim + self.input_dim),
            nn.ReLU(),
            nn.Linear(outlayer_indim, 1)
        )


    def forward(self, x):
        para = x[:, :-2]
        vgvd = x[:, -2:]
        
        # gdnn
        gdnn_out = self.gd_nn(vgvd)
        
        # para_encoder
        para_code = self.para_encoder(para)
        
        # parann
        parann_out = self.para_nn(torch.cat((para, para_code), -1))
        
        # output layer
        y_pred = self.output_layer(torch.cat((gdnn_out, parann_out), -1))
        
        return y_pred
    
    
    def customed_loss(self):

        class my_loss(nn.Module):
            def __init__(self, auto_encoder):
                super().__init__()
                self.encoder = auto_encoder[0]
                self.decoder = auto_encoder[1]

            def forward(self, x, y_pred, y_truth):
                x_para = x[:, :-2]
                x_vgvd = x[:, -2:]

                # encoder-decoder loss
                loss_ae = nn.MSELoss()
                # derivative loss
                loss_deri = nn.MSELoss()
                
                # encoder-decoder
                decoder_output = self.decoder(self.encoder(x_para))

                # 对两个loss求平方和
                return torch.sqrt(loss_ae(decoder_output, x_para)**2 + loss_deri(y_pred, y_truth)**2)

        return my_loss([self.para_encoder, self.para_decoder])



class MyModel5(nn.Module):

    def __init__(self, para_dim, layer_num):
        """
        output[NN([para, AE(para)]), gate(tanhNN(vd), sigNN(vg))]
        """        
        
        super().__init__()
        self.name = 'mine5_AE_sig_tanh_separate_gatenet'
        # AE setting
        self.encode_dim = 4
        self.para_dim = para_dim
        self.para_hidden_dim = (para_dim+2)*2  # assume (input_dim*2, equals(para_dim+2)*2)

        # tanh drain net & sig gate net setting
        self.vgd_hidden_dim = 4  # or less
        self.layer_num = layer_num

        ############## init para ############
        ### AE & para net ###
        self.parann_input_dim = self.encode_dim + self.para_dim

        ### tanh drain net & sig gate net ###
        # Gate subnet 
        self.sig_gate_net = nn.Sequential()

        # Drain subnet
        self.tanh_drain_net = nn.Sequential()
        subnet_input_dim = 1  # [vg] or [vd]

        ############## buid model ############
        # 1. build AE
        self.para_encoder = nn.Sequential(
            nn.Linear(self.para_dim, self.encode_dim*2),
            nn.ReLU(),
            nn.Linear(self.encode_dim*2, self.encode_dim)
        )
        self.para_decoder = nn.Sequential(
            nn.Linear(self.encode_dim, self.encode_dim*2),
            nn.ReLU(),
            nn.Linear(self.encode_dim*2, self.para_dim)
        )
        self.para_nn = nn.Sequential(
            # input
            nn.Linear(self.parann_input_dim, self.para_hidden_dim),
            nn.BatchNorm1d(self.para_hidden_dim),  # change to Layernorm now
            nn.ReLU(),
            
            # hidden layer1
            nn.Linear(self.para_hidden_dim, self.para_hidden_dim),
            nn.BatchNorm1d(self.para_hidden_dim),
            nn.ReLU(),

            # hidden layer2
            nn.Linear(self.para_hidden_dim, self.para_hidden_dim),
            nn.BatchNorm1d(self.para_hidden_dim),
            nn.ReLU(),

            nn.Linear(self.para_hidden_dim, para_dim+2)
        )

        # 2.1 build model-tanh_drain_net
        for id_layer in range(layer_num):
            if id_layer == 0:
                self.tanh_drain_net.add_module(f"layer{id_layer+1}_linear",
                                               nn.Linear(subnet_input_dim, self.vgd_hidden_dim, bias=False))
            else:
                self.tanh_drain_net.add_module(f"layer{id_layer+1}_linear", 
                                               nn.Linear(self.vgd_hidden_dim, self.vgd_hidden_dim, bias=False))
            self.tanh_drain_net.add_module(f"layer{id_layer+1}_tanh",
                                           nn.Tanh())
        # output layer
        self.tanh_drain_net.add_module(f"Output_Layer",
                                       nn.Linear(self.vgd_hidden_dim, self.vgd_hidden_dim, bias=False))
        # TODO：加上激活函数
        
        # 2.2 build model-sig_gate_net
        for id_layer in range(layer_num):
            if id_layer == 0:
                self.sig_gate_net.add_module(f"layer{id_layer+1}_linear",
                                             nn.Linear(subnet_input_dim, self.vgd_hidden_dim))
            else:
                self.sig_gate_net.add_module(f"layer{id_layer+1}_linear", 
                                             nn.Linear(self.vgd_hidden_dim, self.vgd_hidden_dim))
            self.sig_gate_net.add_module(f"layer{id_layer+1}_sig",
                                         nn.Sigmoid())
        # output layer
        self.sig_gate_net.add_module(f"Output_Layer",
                                     nn.Linear(self.vgd_hidden_dim, self.vgd_hidden_dim))
        # TODO：加上激活函数

        # 3. gate net
        self.gate = nn.Sequential(
            nn.Linear(2, 1),
            nn.Sigmoid()
        )

        # 4. output layer
        outlayer_indim = para_dim+2 + self.vgd_hidden_dim  # 改一个：没有gate，output输入【nn, tanhnn, signn】
        self.output_layer = nn.Sequential(
            nn.BatchNorm1d(outlayer_indim),   # 去掉看下
            nn.ReLU(),  # 改成sig或者tanh试下
            nn.Linear(outlayer_indim, 1)
        )


    def forward(self, x):
        """
        output[NN([para, AE(para)]), mix{tanhNN(vd), sigNN(vg)}]
        """
        para = x[:, :-2]
        vg = x[:, -2].unsqueeze(-1)
        vd = x[:, -1].unsqueeze(-1)

        # pass through NN(para_encoder)
        para_code = self.para_encoder(para)
        para_nn_out = self.para_nn(torch.cat((para, para_code), dim=1))

        # pass through tanh_drain_net
        tanh_output = self.tanh_drain_net(vd)

        # pass through sig_gate_net]
        sig_output = self.sig_gate_net(vg)

        # gate
        gate_coe = self.gate(x[:, -2:])
        gate_output = gate_coe * tanh_output + (1-gate_coe) * sig_output

        # output layer
        out_layer_input = torch.cat([para_nn_out, gate_output], dim=1)
        return self.output_layer(out_layer_input)


class MyModel6(nn.Module):

    def __init__(self, para_dim, layer_num):
        """
        output([NN(para, AE(para)), sig(vg), gate(vd)])
        """
        super().__init__()
        self.name = 'mine6_AE_sig_tanh_nogate'
        # AE setting
        self.encode_dim = 4
        self.para_dim = para_dim
        self.para_hidden_dim = (para_dim+2)*2  # assume (input_dim*2, equals(para_dim+2)*2)

        # tanh drain net & sig gate net setting
        self.vgd_hidden_dim = 4  # or less
        self.layer_num = layer_num

        ############## init para ############
        ### AE & para net ###
        self.parann_input_dim = self.encode_dim + self.para_dim

        ### tanh drain net & sig gate net ###
        # Gate subnet 
        self.sig_gate_net = nn.Sequential()

        # Drain subnet
        self.tanh_drain_net = nn.Sequential()
        subnet_input_dim = 1  # [vg] or [vd]

        ############## buid model ############
        # 1. build AE
        self.para_encoder = nn.Sequential(
            nn.Linear(self.para_dim, self.encode_dim*2),
            nn.ReLU(),
            nn.Linear(self.encode_dim*2, self.encode_dim)
        )
        self.para_decoder = nn.Sequential(
            nn.Linear(self.encode_dim, self.encode_dim*2),
            nn.ReLU(),
            nn.Linear(self.encode_dim*2, self.para_dim)
        )
        self.para_nn = nn.Sequential(
            # input
            nn.Linear(self.parann_input_dim, self.para_hidden_dim),
            nn.BatchNorm1d(self.para_hidden_dim),
            nn.ReLU(),
            
            # hidden layer1
            nn.Linear(self.para_hidden_dim, self.para_hidden_dim),
            nn.BatchNorm1d(self.para_hidden_dim),
            nn.ReLU(),

            # hidden layer2
            nn.Linear(self.para_hidden_dim, self.para_hidden_dim),
            nn.BatchNorm1d(self.para_hidden_dim),
            nn.ReLU(),

            nn.Linear(self.para_hidden_dim, para_dim+2)
        )

        # 2.1 build model-tanh_drain_net
        for id_layer in range(layer_num):
            if id_layer == 0:
                self.tanh_drain_net.add_module(f"layer{id_layer+1}_linear",
                                               nn.Linear(subnet_input_dim, self.vgd_hidden_dim))
            else:
                self.tanh_drain_net.add_module(f"layer{id_layer+1}_linear", 
                                               nn.Linear(self.vgd_hidden_dim, self.vgd_hidden_dim))
            self.tanh_drain_net.add_module(f"layer{id_layer+1}_tanh",
                                           nn.Tanh())
        # output layer
        self.tanh_drain_net.add_module(f"Output_Layer",
                                       nn.Linear(self.vgd_hidden_dim, 2))
        # TODO：加上激活函数
        
        # 2.2 build model-sig_gate_net
        for id_layer in range(layer_num):
            if id_layer == 0:
                self.sig_gate_net.add_module(f"layer{id_layer+1}_linear",
                                             nn.Linear(subnet_input_dim, self.vgd_hidden_dim))
            else:
                self.sig_gate_net.add_module(f"layer{id_layer+1}_linear", 
                                             nn.Linear(self.vgd_hidden_dim, self.vgd_hidden_dim))
            self.sig_gate_net.add_module(f"layer{id_layer+1}_sig",
                                         nn.Sigmoid())
        # output layer
        self.sig_gate_net.add_module(f"Output_Layer",
                                     nn.Linear(self.vgd_hidden_dim, 2))
        # TODO：加上激活函数

        # 3. output layer
        outlayer_indim = para_dim+2 + self.vgd_hidden_dim
        self.output_layer = nn.Sequential(
            nn.BatchNorm1d(outlayer_indim),   # 去掉看下
            nn.ReLU(),  # 改成sig或者tanh试下
            nn.Linear(outlayer_indim, 1)
        )


    def forward(self, x):
        """
        output([NN(para, AE(para)), sig(vg), tanh(vd)])
        """
        para = x[:, :-2]
        vg = x[:, -2].unsqueeze(-1)
        vd = x[:, -1].unsqueeze(-1)

        # pass through NN(para_encoder)
        para_code = self.para_encoder(para)
        para_nn_out = self.para_nn(torch.cat((para, para_code), dim=1))

        # pass through tanh_drain_net
        tanh_output = self.tanh_drain_net(vd)

        # pass through sig_gate_net]
        sig_output = self.sig_gate_net(vg)

        # output layer
        out_layer_input = torch.cat([para_nn_out, sig_output, tanh_output], dim=1)
        return self.output_layer(out_layer_input)


class MyModel7(ModelBase):  # 醒醒，狗都不用GP
    
    """
    arch:

    NN[GP(vg, vd), NN[para, AE(para)]]

    output(gd_GP(vg, vd), para_NN(para, encoderNN(para)))
    """
    def __init__(self, input_dim, x, y):
        super().__init__()
        
        self.name = 'mine7_AEGP'
        
        self.input_dim = input_dim
        self.gd_dim = 4
        self.hidden_dim = input_dim * 2
        self.para_dim = input_dim - 2

        self.encode_dim = 4
        
        self.gd_gp = CIGP(x[:, -2:], y)
        
        self.para_encoder = nn.Sequential(
            nn.Linear(self.para_dim, self.encode_dim*2),
            nn.ReLU(),
            nn.Linear(self.encode_dim*2, self.encode_dim)
        )
        self.para_decoder = nn.Sequential(
            nn.Linear(self.encode_dim, self.encode_dim*2),
            nn.ReLU(),
            nn.Linear(self.encode_dim*2, self.para_dim)
        )
        
        self.parann_input_dim = self.encode_dim + self.para_dim
        
        self.para_nn = nn.Sequential(
            # input
            nn.Linear(self.parann_input_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            
            # hidden layer1
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),

            # hidden layer2
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),

            nn.Linear(self.hidden_dim, self.input_dim)
        )
        
        outlayer_indim = 1 + self.input_dim
        self.output_layer = nn.Sequential(
            nn.BatchNorm1d(outlayer_indim),
            nn.ReLU(),
            nn.Linear(outlayer_indim, 1)
        )


    def forward(self, x):
        para = x[:, :-2]
        vgvd = x[:, -2:]
        
        # gdgp
        gdgp_out, gdgp_std = self.gd_gp(vgvd)
        
        # para_encoder
        para_code = self.para_encoder(para)
        
        # parann
        parann_out = self.para_nn(torch.cat((para, para_code), -1))
        
        # output layer
        y_pred = self.output_layer(torch.cat((gdgp_out, parann_out), -1))
        
        return y_pred, gdgp_std


if __name__ == '__main__':
    torch.manual_seed(1)
    x = torch.randn((5, 8))
    y = torch.ones((5, 1))

    model = MyModel7(8, x, y)
    out, std = model(x)

    # model = MyModel2(8)
    # out = model(x)

    print(f"out is {out}")
    print(f"out shape is {out.shape}")
    # print(f"std is {std}")

    n_paras = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"{n_paras} parameters in sum")
