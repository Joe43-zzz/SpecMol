import argparse
import os
import time
import torch
import torch.optim as optim
import numpy as np
import seaborn as sns
from torch_geometric.loader import DataLoader
from model_gnn_pre import LH_Direct, LogReg
from utils_fp_downstream import TestbedDataset
import torch.nn as nn
from train_utils import (
    calculate_auc,
    classification_probe_batch,
    load_model_state_dict,
    ours_loss,
    set_seed,
    train_classification_probe_epoch,
)

from rdkit import RDLogger

# 禁用RDKit的日志输出
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)


def train(spect_net, data_loader, train_optimizer, device, alpha):
    spect_net.train()
    total_loss, total_num, train_bar = 0.0, 0, data_loader

    for tem in train_bar:
        low_x, high_x, spec_x, x_fp = spect_net(tem, device)
        
        loss = ours_loss(low_x, high_x, spec_x, x_fp, alpha)
        #print("loss", loss)

        total_num += len(tem)
        total_loss += loss.item() * len(tem)
        #train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.8f}'.format(epoch, epochs, total_loss / total_num))

        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

    return total_loss / total_num

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DGC Train')
    parser.add_argument('--datafile', default='in-vitro')
    parser.add_argument('--path', default='down_task')
    parser.add_argument('--task', default='bace')
    parser.add_argument('--temperature', default=0.1, type=float)
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--epochs', default=1000, type=int)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--dprate', type=float, default=0.5)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--is_bns', type=bool, default=False)
    parser.add_argument('--act_fn', default='relu', help='activation function')
    parser.add_argument('--K', default=10, type=int)
    parser.add_argument('--random_seed', default=9, type=int)
    parser.add_argument('--hid_dim', default=512, type=int)
    parser.add_argument('--patience', default=100, type=int)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--type', default='tri', type=str) #tri 1489 pub 881 maccs 167 erg 441
    parser.add_argument('--alpha', default=1, type=float)
    parser.add_argument('--split_seed', default=9, type=int)
    parser.add_argument('--use_pair_update', type=int, default=0)
    parser.add_argument('--pair_input_attr_name', type=str, default='edge_attr')
    parser.add_argument('--pair_edge_attr_dim', type=int, default=11)
    parser.add_argument('--pair_update_mode', type=str, default='legacy')
    # args parse
    args = parser.parse_args()
    print(args)
    if args.gpu != -1 and torch.cuda.is_available():
        args.device = "cuda:{}".format(args.gpu)
    else:
        args.device = "cpu"
    
    batch_size, epochs = args.batch_size, args.epochs
    task, random_seed = args.task, args.random_seed
    
    set_seed(random_seed)
    
    clr_tasks = {'bbbp': 1, 'hiv': 1, 'bace': 1, 'tox21': 12, 'clintox': 2, 'sider': 27, 'MUV': 17, 'toxcast':617, 'PCBA':128, 'ecoli':1}
    task_num = clr_tasks[task]
    spec_model = LH_Direct(
        in_dim=93,
        hid_dim=args.hid_dim,
        K=args.K,
        dprate=args.dprate,
        dropout=args.dropout,
        is_bns=args.is_bns,
        act_fn=args.act_fn,
        type=args.type,
    )
    spec_model = spec_model.to(args.device)
    optimizer = optim.Adam(spec_model.parameters(), lr=args.lr, weight_decay=1e-7) #TODO: spec


    datafile = 'now'
    save_name_pre = '{}_{}_{}'.format(batch_size, epochs, datafile)
    if not os.path.exists('results/'+save_name_pre):
        os.mkdir('results/'+save_name_pre)

    data = TestbedDataset(root=args.path, dataset='all', task=task, type=args.type)    
    best_loss = float("inf")
    cnt_wait = 0
    tag = str(int(time.time()))
    best_t = 0
    for epoch in range(0, epochs + 1):
        data_loader = DataLoader(data, batch_size=batch_size, shuffle=True)
        train_loss = train(spec_model, data_loader, optimizer, args.device, args.alpha)

        if train_loss < best_loss:
            best_loss = train_loss
            best_t = epoch
            print(f'In Epoch {epoch}th, the train loss is {best_loss}.')
            torch.save(spec_model.state_dict(), 'pkl/best_spec_model_' + args.task + tag + '.pkl') #TODO: mlp save
            #print('pkl/best_spec_model_' + args.task + tag + '.pkl')
            cnt_wait = 0
        else:
            cnt_wait +=1
        
        if cnt_wait ==args.patience:
            print("Early stopping!")
            break 
        
    print('Loading {}th eppoch'.format( best_t + 1 ))
    
    spec_model.load_state_dict(load_model_state_dict('pkl/best_spec_model_' + args.task + tag + '.pkl'))
    spec_model.eval()
    
    loss_fn = nn.BCEWithLogitsLoss(reduction='mean')

    split_seeds = [9, 19, 29, 39, 49]
    results=[]
    for i in range(5):
        split_seed = split_seeds[i]
        logreg = LogReg(hid_dim=args.hid_dim, n_classes = task_num)
        opt = torch.optim.Adam(logreg.parameters(), lr=0.001, weight_decay=0.0) #original:0.01
        logreg = logreg.to(args.device)
        train_data = TestbedDataset(root=args.path, dataset='train', task=task, type=args.type,seed=split_seed)
        #print("train_data", train_data)
        train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        #print("train_data_loader", train_data_loader)
        val_data = TestbedDataset(root=args.path, dataset='valid', task=task, type=args.type, seed=split_seed)
        val_data_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
        test_data = TestbedDataset(root=args.path, dataset='test', task=task, type=args.type,seed=split_seed)
        test_data_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
        
        val_auc_best = 0
        best_epoch = 0 
        for epoch in range(2000):
            train_classification_probe_epoch(
                train_data_loader, spec_model, logreg, opt, loss_fn, args.device, n_task=task_num
            )
            
            with torch.no_grad():
                val_logits = torch.Tensor().to(args.device)
                val_y = torch.Tensor().to(args.device)
                for tem in val_data_loader:
                    logits, y = classification_probe_batch(tem, spec_model, logreg, args.device, task_num)
                    val_logits = torch.cat((val_logits, logits),0)
                    val_y = torch.cat((val_y, y),0)
                val_auc = calculate_auc(val_y.cpu().numpy(), val_logits.cpu().numpy(), task_num)
                if val_auc>val_auc_best:
                    # print('Val auc in Epoch {} is {}'.format(epoch, val_auc))
                    val_auc_best = val_auc
                    test_logits = torch.Tensor().to(args.device)
                    test_y = torch.Tensor().to(args.device)
                    for tem in test_data_loader:
                        logits, y = classification_probe_batch(tem, spec_model, logreg, args.device, task_num)
                        test_logits = torch.cat((test_logits, logits),0)
                        test_y = torch.cat((test_y, y),0)
                    test_auc = calculate_auc(test_y.cpu().numpy(), test_logits.cpu().numpy(), task_num)
                    # print('Test auc in Epoch {} is {}'.format(epoch, test_auc))
                    best_epoch = epoch
        print('Best Test Auc for {} in {}s Epoch {} is {}'.format(args.task, i, best_epoch, test_auc))
        results.append(float(test_auc))
    results = [v for v in results]
    test_acc_mean = np.mean(results, axis=0) * 100
    values = np.asarray(results, dtype=object)
    uncertainty = np.max(
        np.abs(sns.utils.ci(sns.algorithms.bootstrap(values, func=np.mean, n_boot=1000), 95) - values.mean()))
    print(f'test acc mean = {test_acc_mean:.4f} ± {uncertainty * 100:.4f}')
            
        
