import torch
import torch.nn as nn
"""
refer to the paper 
"Physics-Inspired Neural Networks (Pi-NN) for Efficient Device Compact Modelling" 
for the details of the model.
"""

class PINN(nn.Module):

    def __init__(self, para_dim, hidden_dim, layer_num):
        super().__init__()
        self.name = 'pinn'
        self.para_dim = para_dim
        self.hidden_dim = hidden_dim
        self.layer_num = layer_num
        # layer_num*2+1 funcs in sum, index range from 0 to layer_num*2
        self.funcs_num = layer_num*2+1
        
        # Gate subnet 
        self.sig_gate_net = nn.Sequential()

        # Drain subnet
        self.tanh_drain_net = nn.Sequential()

        subnet_input_dim = para_dim+1  # [para, vg] or [para, vd]
        # build model-tanh_drain_net
        for id_layer in range(layer_num):
            if id_layer == 0:
                self.tanh_drain_net.add_module(f"layer{id_layer+1}_linear",
                                               nn.Linear(subnet_input_dim, hidden_dim, bias=False))
            else:
                self.tanh_drain_net.add_module(f"layer{id_layer+1}_linear", 
                                               nn.Linear(hidden_dim, hidden_dim, bias=False))
            self.tanh_drain_net.add_module(f"layer{id_layer+1}_tanh",
                                           nn.Tanh())
        # output layer
        self.tanh_drain_net.add_module(f"Output_Layer",
                                       nn.Linear(hidden_dim, 1, bias=False))
        # TODO：加上激活函数
        
        # build model-sig_gate_net
        for id_layer in range(layer_num):
            if id_layer == 0:
                self.sig_gate_net.add_module(f"layer{id_layer+1}_linear",
                                             nn.Linear(subnet_input_dim+hidden_dim, hidden_dim))
            else:
                self.sig_gate_net.add_module(f"layer{id_layer+1}_linear", 
                                             nn.Linear(hidden_dim*2, hidden_dim))
            self.sig_gate_net.add_module(f"layer{id_layer+1}_sig",
                                         nn.Sigmoid())
        # output layer
        self.sig_gate_net.add_module(f"Output_Layer",
                                     nn.Linear(1+hidden_dim, 1))
        # TODO：加上激活函数

    def forward(self, x):
        # vg = x[:, -2]
        vd = x[:, -1].unsqueeze(-1)

        # pass through tanh_drain_net 
        tanh_drain_input = torch.cat([x[:, :-2], vd], dim=1)
        tanh_output_cache_list = []
        for id_func in range(self.funcs_num):
            if id_func == 0:
                late_output = self.tanh_drain_net[id_func](tanh_drain_input)
            else:
                late_output = self.tanh_drain_net[id_func](late_output)

            # save tanh() output
            if id_func%2 == 1:
                tanh_output_cache_list.append(late_output)
            # save output of output_layer
            elif id_func == self.funcs_num-1:  # last Linear output layer
                tanh_output_cache_list.append(late_output)  # layer_num+1 cache results in sum
        tanh_drain_output = tanh_output_cache_list[-1]

        # pass through sig_gate_net(cross with drain net)
        for id_layer in range(self.layer_num):  # tanh_output_cache_list has layer_num+1 results
            if id_layer == 0:
                # sig_gate第一层接收[gate_para_vg, tanh_layer0_out]
                gate_para_vg = x[:, :-1]
                tanh_last_layer_output = tanh_output_cache_list[id_layer]
                
                sig_layer0_input = torch.cat([tanh_last_layer_output, gate_para_vg], dim=1)
                # sig(Linear())
                sig_last_layer_output = self.sig_gate_net[1](
                    self.sig_gate_net[0](
                        sig_layer0_input
                    )
                )
            else:
                # sig_gate其他层接收[tanh_last_layer_output(latest), sig_last_layer_output(latest)]
                tanh_last_layer_output = tanh_output_cache_list[id_layer]
                sig_last_layer_input = torch.cat([tanh_last_layer_output, sig_last_layer_output], dim=1)

                # pass through gate
                sig_last_layer_output = self.sig_gate_net[2*id_layer+1](
                    self.sig_gate_net[2*id_layer](
                        sig_last_layer_input
                    ))
        # output of sig_gate_output layer
        sig_output_layer_input = torch.cat([tanh_drain_output, sig_last_layer_output], dim=-1)
        sig_gate_output = self.sig_gate_net[-1](sig_output_layer_input)
        final_output = tanh_drain_output * sig_gate_output

        return final_output



if __name__ == '__main__':
    torch.manual_seed(1)
    x = torch.randn((5, 8))
    print(x)

    model = PINN(6, 4, 3)
    out = model(x)

    torch.save(model, 'tanh_sig_net.pt')

    print(f"out is {out}")

