
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from model_base import ModelBase


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

    ep10
Final loss is 4.126938084026056
Final r2 is 0.9396587075402235
Final mapa is 0.9025257914331091
Final mean-[r2-mapa] is 0.9210922494866662
Final mae is 1.3946313811392088
Final rmse is 2.0314866684342423
    ep20
Final loss is 2.30008968957872
Final r2 is 0.9663696469836086
Final mapa is 0.9243905352157277
Final mean-[r2-mapa] is 0.9453800910996681
Final mae is 1.0846968759397553
Final rmse is 1.5166046583004815
    ep30
Final loss is 2.105117504075227
Final r2 is 0.9692203981767331
Final mapa is 0.9298987315476828
Final mean-[r2-mapa] is 0.9495595648622079
Final mae is 0.9931581173094653
Final rmse is 1.4509023068681182
    ep40
Final loss is 2.0535899554118004
Final r2 is 0.9699737990808247
Final mapa is 0.9328104234553302
Final mean-[r2-mapa] is 0.9513921112680774
Final mae is 0.9705814095532849
Final rmse is 1.4330352247630902
    ep50
Final loss is 1.9320198250760603
Final r2 is 0.9717513151568123
Final mapa is 0.9315695451272648
Final mean-[r2-mapa] is 0.9516604301420386
Final mae is 0.9521247581388249
Final rmse is 1.3899711598001088
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

不好，放弃，应该是gate不好使，二者不是gate权重的关系

LayerNorm:
        ep10
Final loss is 47.32327250414987
Final r2 is 0.3080711733041406
Final mapa is 0.7781369689664694
Final mean-[r2-mapa] is 0.543104071135305
Final mae is 4.069298290952643
Final rmse is 6.879191268176069
        ep20
Final loss is 26.304562664488927
Final r2 is 0.6153925073632334
Final mapa is 0.8559862970269937
Final mean-[r2-mapa] is 0.7356894021951135
Final mae is 2.393012574208868
Final rmse is 5.128797389689801
        ep30
Final loss is 23.72621012784243
Final r2 is 0.6530914311925956
Final mapa is 0.850856088915622
Final mean-[r2-mapa] is 0.7519737600541088
Final mae is 2.3996119024525293
Final rmse is 4.87095577149315

##########
BatchNorm
        ep10

        ep20

        ep30

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
            nn.LayerNorm(self.para_hidden_dim),  # change to Layernorm now
            nn.ReLU(),
            
            # hidden layer1
            nn.Linear(self.para_hidden_dim, self.para_hidden_dim),
            nn.LayerNorm(self.para_hidden_dim),
            nn.ReLU(),

            # hidden layer2
            nn.Linear(self.para_hidden_dim, self.para_hidden_dim),
            nn.LayerNorm(self.para_hidden_dim),
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
            nn.LayerNorm(outlayer_indim),   # 去掉看下
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
        output([NN(para, AE(para)), sig(vg), gate(vd)])  # 不好使，为啥换成两个子网络就不好用了？
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
            nn.LayerNorm(self.para_hidden_dim),  # change to Layernorm now
            nn.ReLU(),
            
            # hidden layer1
            nn.Linear(self.para_hidden_dim, self.para_hidden_dim),
            nn.LayerNorm(self.para_hidden_dim),
            nn.ReLU(),

            # hidden layer2
            nn.Linear(self.para_hidden_dim, self.para_hidden_dim),
            nn.LayerNorm(self.para_hidden_dim),
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
                                       nn.Linear(self.vgd_hidden_dim, 2, bias=False))
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
        outlayer_indim = para_dim+2 + self.vgd_hidden_dim  # 改一个：没有gate，output输入【nn, tanhnn, signn】
        self.output_layer = nn.Sequential(
            nn.LayerNorm(outlayer_indim),   # 去掉看下
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



class MyModel7(nn.Module):

    def __init__(self, para_dim, layer_num):
        super().__init__()
        self.name = 'mine7_AE_sig_tanh_crossed_pinn'   # 不要gate，不好使，目测cross不好使

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
            nn.LayerNorm(self.para_hidden_dim),  # change to Layernorm now
            nn.ReLU(),
            
            # hidden layer1
            nn.Linear(self.para_hidden_dim, self.para_hidden_dim),
            nn.LayerNorm(self.para_hidden_dim),
            nn.ReLU(),

            # hidden layer2
            nn.Linear(self.para_hidden_dim, self.para_hidden_dim),
            nn.LayerNorm(self.para_hidden_dim),
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
            nn.LayerNorm(outlayer_indim),   # 去掉看下
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



if __name__ == '__main__':
    torch.manual_seed(1)
    x = torch.randn((5, 8))
    print(x)

    model = MyModel6(6, 4)
    out = model(x)
    print(f"out is {out}")
    print(f"out shape is {out.shape}")


