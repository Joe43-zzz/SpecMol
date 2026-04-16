import argparse
import os
import time
import torch
import torch.optim as optim
import torch_geometric
from torch_geometric.data import DataLoader
from tqdm import tqdm
from sklearn.manifold import TSNE
from utils_fp_downstream import *
from model_gnn_pre import GNNCon, LH_Direct, LogReg
from nt_xent import NT_Xent
from encoder_gnn import GATNet,GINNet
import random
import torch.nn as nn
from sklearn.metrics import roc_auc_score,r2_score,mean_squared_error

from rdkit import RDLogger

# 禁用RDKit的日志输出
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)


def ours_loss(f_l, f_h, f_a, f_fp, alpha=1, tau=0.5): #TODO: loss + fp contra
    cos = nn.CosineSimilarity(dim=1, eps=1e-6) #TODO dim?eps?
    s_la = torch.exp(cos(f_l, f_a) / tau)
    s_lh = torch.exp(cos(f_l, f_h) / tau)
    s_ha = torch.exp(cos(f_h, f_a) / tau)
    s_fp_a = torch.exp(cos(f_fp, f_a) / tau)
    N = f_l.size(0)
    
    l_denom = torch.sum(s_la) - s_la + s_lh
    h_denom = torch.sum(s_ha) - s_ha + s_lh
    fp_a_denom = torch.sum(s_fp_a)-s_fp_a #+torch.sum(s_fp)-s_fp+torch.sum(s_aa) #sider + bace no
    #fp_denom = torch.sum(s_fp)- s_fp 
    
    loss_l_1 = torch.log(s_la/l_denom)
    loss_h_1 = torch.log(s_ha/h_denom)
    loss_fpa = torch.log(s_fp_a/fp_a_denom)
    
    loss = torch.sum(loss_l_1) + torch.sum(loss_h_1) + alpha*torch.sum(loss_fpa) 
    return -loss/N

def train(spect_net,  data_loader, train_optimizer, device):
    spect_net.train()
    total_loss, total_num, train_bar = 0.0, 0, data_loader

    for tem in train_bar:
        low_x, high_x, spec_x, x_fp = spect_net(tem, args.device)
        
        loss = ours_loss(low_x, high_x, spec_x, x_fp, args.alpha)
        #print("loss", loss)

        total_num += len(tem)
        total_loss += loss.item() * len(tem)
        #train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.8f}'.format(epoch, epochs, total_loss / total_num))

        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

    return total_loss / total_num

def refine_logreg(tem, device, logreg, n_task):
    low_x_mean, high_x_mean, spec_x_mean, x_fp, y = spec_model.get_embedding(tem, device)
    embed = torch.concat([spec_x_mean,x_fp], dim=1)
    logits = logreg(embed)
    y = y.reshape(-1, n_task)
    
    non_999_indices = y != 999
    logits = logits[non_999_indices]
    y = y[non_999_indices]
    return logits, y
    
def calculate_auc(array1,array2,m):
    M = len(array1)
    auc_list = []
    
    for i in range(m):
        sub_array1 = array1[i::m]
        sub_array2 = array2[i::m]
        non_999_indices = sub_array1 != 999
        sub_array1 = sub_array1[non_999_indices]
        sub_array2 = sub_array2[non_999_indices]
        
        if np.unique(sub_array1).size>1:
            auc = roc_auc_score(sub_array1,sub_array2)
            auc_list.append(auc)
           
    return np.mean(auc_list)


def infer_pair_repr_dim(dataset):
    if len(dataset) == 0:
        return None

    sample = dataset[0]
    pair_repr_dim = getattr(sample, 'pair_repr_dim', None)
    if torch.is_tensor(pair_repr_dim):
        pair_repr_dim = int(pair_repr_dim.reshape(-1)[0].item()) if pair_repr_dim.numel() > 0 else None
    elif pair_repr_dim is not None:
        pair_repr_dim = int(pair_repr_dim)

    if pair_repr_dim is not None:
        return pair_repr_dim

    pair_repr_edge = getattr(sample, 'pair_repr_edge', None)
    if torch.is_tensor(pair_repr_edge) and pair_repr_edge.dim() == 2 and pair_repr_edge.size(-1) > 0:
        return int(pair_repr_edge.size(-1))

    pair_repr_full = getattr(sample, 'pair_repr_full', None)
    if torch.is_tensor(pair_repr_full) and pair_repr_full.dim() >= 3 and pair_repr_full.size(-1) > 0:
        return int(pair_repr_full.size(-1))

    return None


def warn_if_pair_repr_batching_is_risky(dataset):
    if len(dataset) == 0:
        return

    sample = dataset[0]
    has_pair_repr_full = hasattr(sample, 'pair_repr_full') and torch.is_tensor(getattr(sample, 'pair_repr_full', None))
    has_pair_repr_edge = hasattr(sample, 'pair_repr_edge') and torch.is_tensor(getattr(sample, 'pair_repr_edge', None))
    if has_pair_repr_full and not has_pair_repr_edge:
        print('Found pair_repr_full without pair_repr_edge. Batched PyG collation may fail for variable-size dense tensors; prefer storing pair_repr_edge as the batching fallback.')

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
    parser.add_argument('--K', default=10)
    parser.add_argument('--random_seed', default=9, type=int)
    parser.add_argument('--hid_dim', default=512, type=int)
    parser.add_argument('--patience', default=100, type=int)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--type', default='tri', type=str) #tri 1489 pub 881 maccs 167 erg 441
    parser.add_argument('--alpha', default=1, type=float)
    parser.add_argument('--split_seed', default=9)
    parser.add_argument('--eval_epochs', default=2000, type=int)
    parser.add_argument('--eval_seeds', default='9,19,29,39,49,59,69,79,89,99')
    # args parse
    args = parser.parse_args()
    print(args)
    if args.gpu != -1 and torch.cuda.is_available():
        args.device = "cuda:{}".format(args.gpu)
    else:
        args.device = "cpu"
    
    split_seed=args.split_seed
    temperature =  args.temperature
    batch_size, epochs = args.batch_size, args.epochs
    task, random_seed = args.task, args.random_seed
    
    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed) # CPU
        torch.cuda.manual_seed(seed) # GPU
        torch.cuda.manual_seed_all(seed) # All GPU
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.backends.cudnn.deterministic = True 
        torch.backends.cudnn.benchmark = False
    
    set_seed(random_seed)
    
    LOG_INTERVAL = 20
    
    clr_tasks = {'bbbp': 1, 'hiv': 1, 'bace': 1, 'tox21': 12, 'clintox': 2, 'sider': 27, 'MUV': 17, 'toxcast':617, 'PCBA':128, 'ecoli':1}
    task_num = clr_tasks[task]
    datafile = 'now'
    save_name_pre = '{}_{}_{}'.format(batch_size, epochs, datafile)
    if not os.path.exists('results/'+save_name_pre):
        os.mkdir('results/'+save_name_pre)

    data = TestbedDataset(root=args.path, dataset='all', task=task, type=args.type)
    pair_repr_dim = infer_pair_repr_dim(data)
    warn_if_pair_repr_batching_is_risky(data)
    if pair_repr_dim is not None:
        print(f'Using Uni-Mol pair repr with dim={pair_repr_dim}.')

    spec_model = LH_Direct(
        in_dim=93,
        hid_dim=args.hid_dim,
        K=args.K,
        dprate=args.dprate,
        dropout=args.dropout,
        is_bns=args.is_bns,
        act_fn=args.act_fn,
        type=args.type,
        pair_repr_dim=pair_repr_dim,
    )
    spec_model = spec_model.to(args.device)
    optimizer = optim.Adam(spec_model.parameters(), lr=args.lr, weight_decay=1e-7) #TODO: spec

    best_loss = float("inf")
    cnt_wait = 0
    tag = str(int(time.time()))
    best_t = 0
    for epoch in range(0, epochs + 1):
        start = time.time()
        data_loader = DataLoader(data, batch_size=batch_size, shuffle=True)
        train_loss = train(spec_model, data_loader, optimizer, args.device)

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
    
    spec_model.load_state_dict(torch.load('pkl/best_spec_model_' + args.task + tag + '.pkl'))
    spec_model.eval()
    
    results = []
    
    logreg = LogReg(hid_dim=args.hid_dim, n_classes = task_num)
    opt = torch.optim.Adam(logreg.parameters(), lr=0.001, weight_decay=0.0) #original:0.01
    logreg = logreg.to(args.device)
    
    loss_fn = nn.BCEWithLogitsLoss(reduction='mean')
    
    split_seeds = [int(seed.strip()) for seed in str(args.eval_seeds).split(',') if seed.strip()]
    for split_seed in split_seeds:
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
        for epoch in range(args.eval_epochs):
            logreg.train()
            opt.zero_grad()
            for tem in train_data_loader:
                logits, y = refine_logreg(tem, args.device, logreg, n_task=task_num)
                loss = loss_fn(logits, y)
                #print(f"Loss in epoch {epoch} is {loss.item()}")
                loss.backward()
            opt.step()
            
            with torch.no_grad():
                val_logits = torch.Tensor().to(args.device)
                val_y = torch.Tensor().to(args.device)
                for tem in val_data_loader:
                    logits, y = refine_logreg(tem, args.device, logreg, task_num)
                    val_logits = torch.cat((val_logits, logits),0)
                    val_y = torch.cat((val_y, y),0)
                val_auc = calculate_auc(val_y.cpu().numpy(), val_logits.cpu().numpy(), task_num)
                if val_auc>val_auc_best:
                    print('Val auc in Epoch {} is {}'.format(epoch, val_auc))
                    val_auc_best = val_auc
                    test_logits = torch.Tensor().to(args.device)
                    test_y = torch.Tensor().to(args.device)
                    for tem in test_data_loader:
                        logits, y = refine_logreg(tem, args.device, logreg, task_num)
                        test_logits = torch.cat((test_logits, logits),0)
                        test_y = torch.cat((test_y, y),0)
                    test_auc = calculate_auc(test_y.cpu().numpy(), test_logits.cpu().numpy(), task_num)
                    print('Test auc in Epoch {} is {}'.format(epoch, test_auc))
                    best_epoch = epoch
        print('Best Test Auc for {} in Epoch {} is {}'.format(args.task, best_epoch, test_auc))
            
        
