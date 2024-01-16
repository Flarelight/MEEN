import torch
import torch.nn as nn
from model_base import ModelBase
from tensorboardX import SummaryWriter


class MEEN(ModelBase):
    
    """
    arch:

    NN[NN(vg, vd), NN[para, AE(para)]]

    output(gd_NN(vg, vd), para_NN(para, encoderNN(para)))
    """
    def __init__(self, input_dim):
        super(MEEN, self).__init__()
        
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
        
        # self.gd_nn = nn.Sequential(
        #     # input
        #     nn.Linear(2, self.gd_dim),
        #     nn.BatchNorm1d(self.gd_dim),
        #     # nn.LogSigmoid(),
        #     nn.ReLU(),
            
        #     # hidden layer1
        #     nn.Linear(self.gd_dim, self.gd_dim),
        #     nn.BatchNorm1d(self.gd_dim),
        #     # nn.LogSigmoid(),
        #     nn.ReLU(),

        #     # hidden layer2
        #     nn.Linear(self.gd_dim, self.gd_dim),
        #     nn.BatchNorm1d(self.gd_dim),
        #     # nn.LogSigmoid(),
        #     nn.ReLU(),
            
        #     # hidden layer3
        #     nn.Linear(self.gd_dim, self.gd_dim),
        #     nn.BatchNorm1d(self.gd_dim),
        #     # nn.LogSigmoid(),
        #     nn.ReLU(),
            
        #     nn.Linear(self.gd_dim, self.gd_dim)
        # )
        self.tanh_drain_nn = nn.Sequential(
            nn.Linear(1, self.gd_dim),
            nn.BatchNorm1d(self.gd_dim),
            nn.Tanh(),
            
            nn.Linear(self.gd_dim, self.gd_dim),
            nn.BatchNorm1d(self.gd_dim),
            nn.Tanh(),
            
            nn.Linear(self.gd_dim, self.gd_dim),
            nn.BatchNorm1d(self.gd_dim),
            nn.Tanh(),
            
            nn.Linear(self.gd_dim, self.gd_dim)
        )
        self.sig_gate_nn = nn.Sequential(
            nn.Linear(1, self.gd_dim),
            nn.BatchNorm1d(self.gd_dim),
            nn.Sigmoid(),
            
            nn.Linear(self.gd_dim, self.gd_dim),
            nn.BatchNorm1d(self.gd_dim),
            nn.Sigmoid(),
            
            nn.Linear(self.gd_dim, self.gd_dim),
            nn.BatchNorm1d(self.gd_dim),
            nn.Sigmoid(),
            
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
        
        outlayer_indim = self.gd_dim*2 + self.input_dim
        self.output_layer = nn.Sequential(
            nn.BatchNorm1d(outlayer_indim),
            nn.ReLU(),
            nn.Linear(outlayer_indim, 1)
        )


    def forward(self, x):
        para = x[:, :-2]
        vgvd = x[:, -2:]
        vg = x[:, -2]
        vd = x[:, -1]
        
        vg.unsqueeze_(1)
        vd.unsqueeze_(1)
        # gdnn
        tanh_drain_out = self.tanh_drain_nn(vd)
        sig_gate_out = self.sig_gate_nn(vg)
        expert_vg_vd = torch.cat([tanh_drain_out * sig_gate_out, sig_gate_out], dim=1)
        
        # para_encoder
        para_code = self.para_encoder(para)
        
        # parann
        parann_out = self.para_nn(torch.cat((para, para_code), -1))
        
        # output layer
        y_pred = self.output_layer(torch.cat((expert_vg_vd, parann_out), -1))
        
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



if __name__ == '__main__':
    torch.manual_seed(1)
    x = torch.randn((5, 8))
    y = torch.ones((5, 1))

    model = MEEN(8)
    out = model(x)

    print(f"out is {out}")
    print(f"out shape is {out.shape}")

    # torch.save(model, 'MEEN.pt')
    tb_writer = SummaryWriter('./paper_graph')
    tb_writer.add_graph(model, x)
