# ---- coding: utf-8 ----
# @Author: Wu Yinpeng
# @Version: v01
# @Contact: YP_Wu@buaa.edu.cn
# @Date: 2023/1/23

"""
intro of this file
"""
import sys
import os
from datetime import datetime
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.nn import RNN, LSTM, GRU
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from tensorboardX import SummaryWriter
from math import sqrt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score 
import argparse
from tools.data_process import get_data
from tools.utils import get_project_root_path, path_join
from models.ann import ANNX, ANN
from models.rtt_device import RTTDeviceModel
from models.pinn import PINN
from models.mine import MyModel1, MyModel2, MyModel3, MyModel4, MyModel5, MyModel6, MyModel7


# DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
# BATCH_SIZE = 225  # 121(11x11) for gaa; 225(15x15) for planar
# LR = 0.001
# EPOCH = 10  # 500, 1000

class MAPA(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, y_truth, y_pred):
        y_truth, y_pred = np.array(y_truth), np.array(y_pred)
        # Avoid division by zero
        return 1 - np.mean(np.abs((y_truth - y_pred) / y_truth))


def eval_model(model, loss_fnc, te_loader, device, is_final: bool = False, vg_h=None, vd_h=None, aux_h=None):
    model.eval()
    total_loss = 0
    y_truth_list, y_pred_list = [], []

    for idx, (X, y) in enumerate(te_loader):
        X, y = X.to(device, non_blocking=False), y.to(device, non_blocking=False)
        
        # if model.name == 'mine':
        #     if model.mode == 'base':
        #         y_pred, _, _ = model(X, vg_h, vd_h)
                
        #     elif model.mode == 'res' or 'moe':
        #         y_pred, _, _, _ = model(X, vg_h, vd_h, aux_h)
        if model.name == 'mine2_AE':
            y_pred = model(X)
            # loss = model.eval_loss(y_pred, y)
            loss = loss_fnc(y_pred, y)
        
        elif model.name == 'rnn':
            X.unsqueeze_(1)
            y_pred, _ = model(X)
            y_pred.squeeze_(1)
            loss = loss_fnc(y_pred, y)
        
        elif model.name == 'mine7_AEGP':
            y_pred, y_std = model(X)
            loss = model.gd_gp.negative_log_likelihood() + loss_fnc(y_pred, y)

        else:
            y_pred = model(X)
            # loss = model.eval_loss(y_pred, y)
            loss = loss_fnc(y_pred, y)

        y_truth_list += list(y.cpu().detach().numpy())
        y_pred_list += list(y_pred.cpu().detach().numpy())

        total_loss += loss.item()
        
    eval_r2 = r2_score(y_truth_list, y_pred_list)
    eval_mae = mean_absolute_error(y_truth_list, y_pred_list)
    eval_mapa = MAPA()(y_truth_list, y_pred_list)
    eval_rmse = sqrt(mean_squared_error(y_truth_list, y_pred_list))
    
    if is_final:
        x_te, y_te = te_loader.dataset.tensors
        x_te, y_te = x_te.to(device, non_blocking=False), y_te.to(device, non_blocking=False)
        # if model.name == 'mine':
        #     if model.mode == 'base':
        #         y_pred, _, _ = model(x_te, vg_h, vd_h)
            
        #     elif model.mode == 'res' or 'moe':
        #         y_pred, _, _, _ = model(x_te, vg_h, vd_h, aux_h)
        if model.name == 'mine2_AE':
            y_pred = model(x_te)
            # loss = model.eval_loss(y_pred, y_te).item()
        elif model.name == 'rnn':
            x_te.unsqueeze_(1)
            y_pred, _ = model(x_te)
            y_pred.squeeze_(1)
        elif model.name == 'mine7_AEGP':
            y_pred, y_std = model(x_te)
        else:
            y_pred = model(x_te)
            # loss = model.eval_loss(y_pred, y_te).item()
        
        loss = loss_fnc(y_pred, y_te).item()

        # move to cpu
        y_te, y_pred = y_te.cpu().detach(), y_pred.cpu().detach()
        
        eval_r2 = r2_score(y_te, y_pred)
        eval_mae = mean_absolute_error(y_te, y_pred)
        eval_mapa = MAPA()(y_te, y_pred)
        eval_rmse = sqrt(mean_squared_error(y_te, y_pred))
        
        if model.name == 'mine7_AEGP':
            return loss, eval_r2, eval_mae, eval_mapa, eval_rmse, y_pred, y_std.detach()
        else:
            return loss, eval_r2, eval_mae, eval_mapa, eval_rmse, y_pred
    
    else:
        return total_loss, eval_r2, eval_mae, eval_mapa, eval_rmse


def train_model(model, opt, tr_loader, te_loader, device, args):

    model.train()
    total_loss, total_step = 0, 0
    optimizer = None

    result_save_dir = f"results/{args['device']}/{args['dataset']}_{args['model']}_{model.name}_seed{args['seed']}_epoch{args['epoch']}_lr{args['lr']}_bz{args['batch_size']}"
    
    tb_writer = SummaryWriter(result_save_dir)

    if opt == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=float(args['lr']))
    elif opt == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=float(args['lr']))
    # opt_sche = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=30, verbose=True)
    
    # declare scheduler for mine model
    if model.name in ['mine2_AE', 'mine6_AE_sig_tanh_nogate']:
        opt_sche = CosineAnnealingLR(optimizer, T_max=args['epoch'], eta_min=1e-6)  # planar: 1e-8
    
    # # 如果是涉及到AE的部分，需要先针对输入训练编码器
    if model.name in ['mine2_AE', 'mine5_AE_sig_tanh_crossed_pinn', 'mine6_AE_sig_tanh_nogate']:
        # check if weights exist
        if args['dataset'] == 'planar':
            model.para_encoder.load_state_dict(torch.load('results/mine2_encoder_weigths.pt'))
            model.para_decoder.load_state_dict(torch.load('results/mine2_decoder_weigths.pt'))
            print(f"AE loads pre-trained weights.")
        elif args['dataset'] == 'gaa':
            model.para_encoder.load_state_dict(torch.load('results/gaa_mine2_encoder_weigths.pt'))
            model.para_decoder.load_state_dict(torch.load('results/gaa_mine2_decoder_weigths.pt'))
            print(f"AE loads pre-trained weights.")
        # train if not exists
        else:
            thresh = 1.47  # planar 0.24
            para = tr_loader.dataset.tensors[0][:, :-2]
            para_opt = torch.optim.Adam([{'params': model.para_encoder.parameters()},
                                         {'params': model.para_decoder.parameters()}],
                                         lr=1e-2)
            para_sche = ReduceLROnPlateau(para_opt, mode='min', factor=0.1, patience=10, verbose=True)
            ae_loss_fnc = nn.MSELoss()
            encoder_decoder_loss = 1
            ae_steps = 0
            while encoder_decoder_loss > thresh:
                para_opt.zero_grad()

                # encoder-decoder
                decoder_output = model.para_decoder(model.para_encoder(para))
                # 优化en-decoder的损失
                encoder_decoder_loss = ae_loss_fnc(decoder_output, para)
                encoder_decoder_loss.backward()
                # 优化器更新参数
                para_opt.step()
                encoder_decoder_loss = encoder_decoder_loss.item()
                # 更新调度器的lr
                para_sche.step(encoder_decoder_loss)

                ae_steps += 1
                tb_writer.add_scalar("AE loss", encoder_decoder_loss, ae_steps)
                print(f"AE Loss: {encoder_decoder_loss:>4f}\t\t [{ae_steps:>5d}]")
                # if ae_steps%1000==0:
                #     thresh += 0.05

            # save weights
            torch.save(model.para_encoder.state_dict(), 'results/gaa_mine2_encoder_weigths.pt')
            torch.save(model.para_decoder.state_dict(), 'results/gaa_mine2_decoder_weigths.pt')
        
    for e in range(args['epoch']):
        y_truth_list, y_pred_list = [], []

        print(f"Epoch {e + 1}\n----------------------------------------------")
        size = len(tr_loader.dataset)
        
        # # init h0 for rtt_device, rnn
        # h_last = None
        # # init h0 for mine model
        # vg_t_h, vd_t_h, aux_t_h = None, None, None
        # rnn h0
        rnn_h0 = None

        for batch, (X, y) in enumerate(tr_loader):
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            
            if model.name == 'rtt_device':
                y_pred, h_last = model(X, h_last)
                # customed loss api
                loss = model.train_loss(y_pred, y) / y.shape[0]
            
            # elif model.name == 'mine':
            #     # if model.mode == 'base':
            #     #     y_pred, vg_t_h, vd_t_h = model(X, vg_t_h, vd_t_h)
                    
            #     # elif model.mode == 'res' or 'moe':
            #     #     y_pred, vg_t_h, vd_t_h, aux_t_h = model(X, vg_t_h, vd_t_h, aux_t_h)
                    
            #     y_pred = model(X)
            #     # customed loss api
            #     loss = model.loss_func(y_pred, y) / y.shape[0]
            elif model.name == 'rnn':
                X.unsqueeze_(1)
                y_pred, rnn_h0 = model(X)
                y_pred.squeeze_(1)

                loss = nn.MSELoss()(y_pred, y)
                loss = loss / y.shape[0]
        
            elif model.name == 'pinn':
                y_pred = model(X)
                loss = nn.MSELoss()(y_pred, y)
                loss = loss / y.shape[0]

            elif model.name == 'mine2_AE':
                # X.requires_grad_(True)
                y_pred = model(X)
                
                # # derivative loss 
                # y_pred.sum().backward(retain_graph=True)
                # derivative_loss = torch.norm(X.grad[:, -2:], p=2)

                # loss1: MSEloss of pred & truth; loss2: L2(dId/dvg, dId/dvd)
                # loss = nn.MSELoss()(y_pred, y) + derivative_loss
                # loss = nn.MSELoss()(y_pred, y) + model.loss_lambda*derivative_loss
                loss = nn.MSELoss()(y_pred, y)
                loss = loss / y.shape[0]

            elif model.name == 'mine3_simp':
                pass
            elif model.name == 'mine4_lstm':
                pass
            elif model.name == 'mine7_AEGP':
                y_pred, _ = model(X)
                loss1 = model.gd_gp.negative_log_likelihood()
                loss = loss1 + nn.MSELoss()(y_pred, y)
                loss = loss / y.shape[0]
            else:
                y_pred = model(X)
                loss = nn.MSELoss()(y_pred, y) / y.shape[0]
            
            y_truth_list += list(y.cpu().detach().numpy())
            y_pred_list += list(y_pred.cpu().detach().numpy())
            
            # BP
            loss.backward(retain_graph=True)

            optimizer.step()
            # 换用调度器控制优化器的lr
            # opt_sche.step(loss)
            if model.name in ['mine2_AE', 'mine6_AE_sig_tanh_nogate']:
                opt_sche.step()

            total_loss += loss.item()
            total_step += 1

            if total_step % 50 == 0:
                loss, cur = loss.item(), batch * len(X)
                
                r2 = r2_score(y.cpu().detach(), y_pred.cpu().detach())
                mae = mean_absolute_error(y.cpu().detach(), y_pred.cpu().detach())
                mapa = MAPA()(y.cpu().detach(), y_pred.cpu().detach())
                mean_r2_mapa = (r2+mapa)/2
                rmse = sqrt(mean_squared_error(y.cpu().detach(), y_pred.cpu().detach()))
                
                print(f"Loss: {loss:>5f}\t\t [{cur:>5d} / {size:>5d}]")
                tb_writer.add_scalar("Train loss", total_loss / total_step, total_step)
                tb_writer.add_scalar("Train R2", r2, total_step)
                tb_writer.add_scalar("Train MAPA", mapa, total_step)
                tb_writer.add_scalar("Train MEAN_R2_MAPA", mean_r2_mapa, total_step)
                tb_writer.add_scalar("Train MAE", mae, total_step)
                tb_writer.add_scalar("Train RMSE", rmse, total_step)

                tb_writer.add_scalar("Current lr", optimizer.param_groups[0]['lr'])
                print(f"Current lr is {optimizer.param_groups[0]['lr']}")

        eval_loss, eval_r2, eval_mae, eval_mapa, eval_rmse = eval_model(model, nn.MSELoss(), te_loader, device)
        eval_mean_r2_mapa = (eval_r2+eval_mapa)/2

        tb_writer.add_scalar("Eval loss", eval_loss / total_step, total_step)
        tb_writer.add_scalar("Eval R2", eval_r2, total_step)
        tb_writer.add_scalar("Eval MAPA", eval_mapa, total_step)
        tb_writer.add_scalar("Eval MEAN_R2_MAPA", eval_mean_r2_mapa, total_step)
        tb_writer.add_scalar("Eval MAE", eval_mae, total_step)
        tb_writer.add_scalar("Eval RMSE", eval_rmse, total_step)

        print(f"Epoch {e+1}, \n \
              val loss is {eval_loss / len(te_loader)}; \n \
              val r2 is {eval_r2}; \n \
              val mapa is {eval_mapa}; \n \
              val mean_r2_mapa is {eval_mean_r2_mapa}; \n \
              val mae is {eval_mae}; \n \
              val rmse is {eval_rmse}") 

    print(f"Model: {args['model']}")
    print(f"Result saved to {result_save_dir}")
    return result_save_dir


def main():

    # arguments
    cur_pth = path_join(get_project_root_path(), "src/config.yaml")
    args = yaml.load(open(cur_pth, 'rb'), Loader=yaml.FullLoader)

    # output redirection
    formatted_now = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    output_file = path_join(get_project_root_path(), f"results/{formatted_now}.txt")
    sys.stdout = open(output_file, 'w')

    print("Current Timestamp: ", formatted_now)
    print("--------------------------CASE-------------------------")
    print(args)

    # fix seed 
    torch.manual_seed(args['seed'])
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    # check argument `device`
    device = torch.device('cpu')
    # if args['device'] == 'gpu':
    #     check_gpu_flag = torch.cuda.is_available()
    #     print(f"Detect GPU Devices: {check_gpu_flag}")
    #     if check_gpu_flag:
    #         print(f"Using GPU0...")
    #         device = torch.device('cuda:0')
    #     else:
    #         print(f"Using CPU...")

    # get Dataloader
    x_tr, x_te, y_tr, y_te = get_data(args['dataset'])
    x_dim = x_tr.shape[1]

    # x_tr.requires_grad_(True)
    tr_loader = DataLoader(TensorDataset(x_tr, y_tr), batch_size=args['batch_size'], shuffle=True)
    te_loader = DataLoader(TensorDataset(x_te, y_te), batch_size=args['batch_size'], shuffle=False)

    # init model
    model = None
    if args['model'] == 'ann':
        model = ANNX(input_dim=x_tr.shape[1],
                     output_dim=y_tr.shape[1])
        
    elif args['model']== 'rtt_device':
        model = RTTDeviceModel(input_dim=x_tr.shape[1],
                               hidden_dim=8,
                               num_layers=4,
                               activation='tanh',
                               dropout=0.3,
                               conv=args['conv'])
    elif args['model'] == 'rnn':
        # input_dim=x_dim
        # hidden_dim=1
        # hidden_layers=3
        model = RNN(x_dim, 1, 3)
        model.name = 'rnn'
    
    elif args['model'] == 'pinn':
        model = PINN(x_dim-2, 4, 3)
    
    elif args['model'] == 'MEEN':
        # model = MyModel1(input_dim=x_tr.shape[1],
        #                  hidden_dim=8,
        #                  num_layers=4,
        #                  dropout=0.3,
        #                  mode=args.mymode)
        model = MyModel2(input_dim=x_tr.shape[1])
    elif args['model'] == 'mine3':
        # model = MyModel1(input_dim=x_tr.shape[1],
        #                  hidden_dim=8,
        #                  num_layers=4,
        #                  dropout=0.3,
        #                  mode=args.mymode)
        model = MyModel3(input_dim=x_tr.shape[1])
    
    elif args['model'] == 'mine5':
        model = MyModel5(para_dim=x_dim-2,
                         layer_num=4)
        
    elif args['model'] == 'mine6':
        model = MyModel6(para_dim=x_dim-2,
                         layer_num=4)
    
    elif args['model'] == 'mine7':
        model = MyModel7(input_dim=x_dim,
                         x=x_tr,
                         y=y_tr)
    
    else:
        raise NotImplementedError("This model is not be implemented")
    
    n_paras = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{n_paras} parameters in sum")
    print("########################################################")

    # train
    print(f"Detect GPU Devices: {torch.cuda.is_available()}")
    model = model.to(device)
    
    result_dir = train_model(model=model,
                             opt=args['opt'],
                             tr_loader=tr_loader,
                             te_loader=te_loader,
                             device=device,
                             args=args)
    
    # final evaluate
    final_loss, final_r2, final_mae, final_mapa, final_rmse, y_pred = eval_model(model=model,
                                                                                 loss_fnc=nn.MSELoss(),
                                                                                 te_loader=te_loader,
                                                                                 device=device,
                                                                                 is_final=True)
    mean_r2_mapa = (final_mapa+final_r2)/2

    # print
    print(f"Final loss is {final_loss}")
    print(f"Final r2 is {final_r2}")
    print(f"Final mapa is {final_mapa}")
    print(f"Final mean-[r2-mapa] is {mean_r2_mapa}")
    print(f"Final mae is {final_mae}")
    print(f"Final rmse is {final_rmse}")
    print(f"Final y_pred is {y_pred}")
    print(f"y_te is {y_te}")

    # save final y_pred & metrix to result_dir
    torch.save(final_loss, result_dir + '/loss.pt')
    torch.save(final_r2, result_dir + '/r2.pt')
    torch.save(final_mapa, result_dir + '/mapa.pt')
    torch.save(mean_r2_mapa, result_dir + '/mean_r2_mapa.pt')
    torch.save(final_mae, result_dir + '/mae.pt')
    torch.save(final_rmse, result_dir + '/rmse.pt')
    torch.save(y_pred, result_dir + '/y_pred.pt')

    # 关闭文件并恢复标准输出
    sys.stdout.close()
    sys.stdout = sys.__stdout__

if __name__ == '__main__':

    main()

