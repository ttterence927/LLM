from data_provider.data_factory import data_provider
from utils.tools import EarlyStopping, adjust_learning_rate, visual, vali, test
from tqdm import tqdm
from models.PatchTST import PatchTST
from models.GPT4TS import GPT4TS
from models.DLinear import DLinear
from models.TimesNet import Model

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from loss.dilate_loss import dilate_loss
from models.optimizers import get_optimizer

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np

import argparse
import random
import torch.nn.functional as F
from models.loss import NoFussCrossEntropyLoss, BinaryLoss, l2_reg_loss
warnings.filterwarnings('ignore')
alpha = 0.55
gamma  = 0.02
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

if __name__ == "__main__":
    
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='GPT4TS')

    parser.add_argument('--model_id', type=str, required=True, default='test')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/')

    parser.add_argument('--root_path', type=str, default='./dataset/traffic/')
    parser.add_argument('--data_path', type=str, default='traffic.csv')
    parser.add_argument('--data', type=str, default='custom')
    parser.add_argument('--features', type=str, default='M')
    parser.add_argument('--freq', type=int, default=1)
    parser.add_argument('--target', type=str, default='y')
    parser.add_argument('--embed', type=str, default='timeF')
    parser.add_argument('--percent', type=int, default=10)

    parser.add_argument('--seq_len', type=int, default=512)
    parser.add_argument('--pred_len', type=int, default=96)
    parser.add_argument('--label_len', type=int, default=48)

    parser.add_argument('--decay_fac', type=float, default=0.75)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--train_epochs', type=int, default=10)
    parser.add_argument('--lradj', type=str, default='type1')
    parser.add_argument('--patience', type=int, default=3)

    parser.add_argument('--gpt_layers', type=int, default=3)
    parser.add_argument('--is_gpt', type=int, default=1)
    parser.add_argument('--e_layers', type=int, default=3)
    parser.add_argument('--d_model', type=int, default=768)
    parser.add_argument('--n_heads', type=int, default=16)
    parser.add_argument('--d_ff', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--enc_in', type=int, default=862)
    parser.add_argument('--c_out', type=int, default=862)
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--kernel_size', type=int, default=25)
    parser.add_argument('--weight_decay',type=float,default = 0.01)

    parser.add_argument('--loss_func', type=str, default='mse')
    parser.add_argument('--pretrain', type=int, default=1)
    parser.add_argument('--freeze', type=int, default=1)
    parser.add_argument('--model', type=str, default='model')
    parser.add_argument('--stride', type=int, default=8)
    parser.add_argument('--max_len', type=int, default=-1)
    parser.add_argument('--hid_dim', type=int, default=16)
    parser.add_argument('--tmax', type=int, default=10)
    parser.add_argument('--feature_num', type=int, default=112)
    parser.add_argument('--itr', type=int, default=3)
    parser.add_argument('--cos', type=int, default=0)
    parser.add_argument('--class_ratio', type=float, default=0.5)
    parser.add_argument('--l2_reg', type=float, default=0.1)
    parser.add_argument('--optimizer', type=str, default='RAdam')


    args = parser.parse_args()

    SEASONALITY_MAP = {
    "minutely": 1440,
    "10_minutes": 144,
    "half_hourly": 48,
    "hourly": 24,
    "daily": 7,
    "weekly": 1,
    "monthly": 12,
    "quarterly": 4,
    "yearly": 1
    }

    mses = []
    maes = []

    for ii in range(args.itr):

        setting = '{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_gl{}_df{}_eb{}_itr{}'.format(args.model_id, 336, args.label_len, args.pred_len,
                                                                        args.d_model, args.n_heads, args.e_layers, args.gpt_layers, 
                                                                        args.d_ff, args.embed, ii)
        path = os.path.join(args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        if args.freq == 0:
            args.freq = 'h'

        train_data, train_loader = data_provider(args, 'train')
        vali_data, vali_loader = data_provider(args, 'val')
        test_data, test_loader = data_provider(args, 'test')
        

        if args.freq != 'h':
            args.freq = SEASONALITY_MAP[test_data.freq]
            print("freq = {}".format(args.freq))

        device = torch.device('cuda:0')

        time_now = time.time()
        train_steps = len(train_loader)

        if args.model == 'PatchTST':
            args.feature_num = train_data.data_x.shape[1]
            print(args.feature_num)
            model = PatchTST(args, device)
            print(model)
            model.to(device)
        elif args.model == 'DLinear':
            model = DLinear(args, device)
            model.to(device)
        elif args.model == 'TimesNet':
            model = Model(args)
            model.to(device)
        else:
            model = GPT4TS(args, device)
        # mse, mae = test(model, test_data, test_loader, args, device, ii)

        params = model.parameters()
        #model_optim = torch.optim.Adam(params, lr=args.learning_rate)
        optim_class = get_optimizer(args.optimizer)
        model_optim = optim_class(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        early_stopping = EarlyStopping(patience=args.patience, verbose=True)
        print('loss using: ',args.loss_func)
        if args.loss_func == 'mse':

            criterion = nn.MSELoss()
        elif args.loss_func == 'dilate':
            class DilateLoss(nn.Module):
                def __init__(self):
                    super(DilateLoss, self).__init__()

                def forward(self, pred, true):
                    loss, _, _ = dilate_loss(pred, true,alpha, gamma, device)
                    return loss
            criterion = DilateLoss()
        elif args.loss_func == 'smape':
            class SMAPE(nn.Module):
                def __init__(self):
                    super(SMAPE, self).__init__()
                def forward(self, pred, true):
                    return torch.mean(200 * torch.abs(pred - true) / (torch.abs(pred) + torch.abs(true) + 1e-8))
            criterion = SMAPE()
        elif args.loss_func == 'tildeq':
            from loss.dilate_loss import dilate_loss, tildeq_loss
            criterion = lambda x,y: tildeq_loss(x,y, alpha = alpha, gamma = gamma)
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=args.tmax, eta_min=1e-8)
        if args.num_classes == 2: 
            criterion_classification = BinaryLoss() #nn.CrossEntropyLoss()
        else:
            criterion_classification = NoFussCrossEntropyLoss()
        for epoch in range(args.train_epochs):

            iter_count = 0
            train_loss = []
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(train_loader)):

                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(device)

                batch_y = batch_y.float().to(device)
                batch_x_mark = batch_x_mark.float().to(device)
                batch_y_mark = batch_y_mark.float().to(device)
                #print(batch_y.shape)
                #print(batch_y_mark.shape)
                batch_y_reg = batch_y  # Regression targets
                batch_y_cls = batch_y_mark   # Classification targets, assuming last column is class label
                #print(batch_x.shape)
                regression_output, classification_output = model(batch_x, iter_count)
                #print(regression_output.shape)
                regression_output = regression_output.squeeze()#[:, -args.pred_len:]
                #classification_output = classification_output[:, -args.num_classes:]
                #print(regression_output.shape)
                #print(classification_output.shape)
                #classification_output = classification_output[:, -args.pred_len:, :].reshape(-1, classification_output.shape[-1])

                batch_y_reg = batch_y_reg[:, -args.pred_len:].squeeze().to(device)
                batch_y_cls = batch_y_cls.squeeze().to(device)  # Reshape and convert to long for classification criterion
                
                #print(classification_output.shape)
                #print(regression_output.unsqueeze(-1).shape, batch_y_reg.unsqueeze(-1).shape)
                #print(classification_output.shape, batch_y_cls.shape)
                loss_reg = criterion(regression_output.unsqueeze(-1), batch_y_reg.unsqueeze(-1))
                loss_cls = criterion_classification(classification_output, batch_y_cls.unsqueeze(-1))
                #print(loss_reg.cpu(), loss_cls.cpu())
                loss = (1- args.class_ratio) * loss_reg + args.class_ratio * loss_cls  # Combine losses, you can also weigh them differently

                train_loss.append(loss.item())

                if (i + 1) % 1000 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

            
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

            train_loss = np.average(train_loss)

            vali_loss = vali(model, vali_data, vali_loader, criterion, criterion_classification, args, device, ii)
            # test_loss = vali(model, test_data, test_loader, criterion, args, device, ii)
            # print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}, Test Loss: {4:.7f}".format(
            #     epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss))

            if args.cos:
                scheduler.step()
                print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
            else:
                adjust_learning_rate(model_optim, epoch + 1, args)
            early_stopping(vali_loss, model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        best_model_path = path + '/' + 'checkpoint.pth'
        model.load_state_dict(torch.load(best_model_path))
        print("------------------------------------")
        #mse, mae = test(model, test_data, test_loader, criterion, criterion_classification, args, device, ii)
        #mses.append(mse)
        #maes.append(mae)

    #mses = np.array(mses)
    #maes = np.array(maes)
    #print("mse_mean = {:.4f}, mse_std = {:.4f}".format(np.mean(mses), np.std(mses)))
    #print("mae_mean = {:.4f}, mae_std = {:.4f}".format(np.mean(maes), np.std(maes)))