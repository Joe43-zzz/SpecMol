import random
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.utils import get_laplacian 
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool as gmp
from LH_Direct_ChebnetII_prop import ChebnetII_prop

class LogReg(nn.Module):
    def __init__(self, hid_dim, n_classes, dropout=0.2):
        super(LogReg, self).__init__()

        self.fc = nn.Linear(hid_dim*2, hid_dim)
        self.fc1 = nn.Linear(hid_dim, n_classes)
        self.dp = nn.Dropout(dropout)

    def forward(self, x):
        ret = self.fc(x)
        ret = self.dp(ret)
        ret = self.fc1(ret)
        return ret


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0.2):
        super(MLP, self).__init__()
        
        # 定义三层全连接层
        self.fc1 = nn.Linear(input_size, hidden_size)  
        self.fc2 = nn.Linear(hidden_size, hidden_size) 
        self.fc3 = nn.Linear(hidden_size, output_size) 
        
        # dropout 防止过拟合
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))       
        x = self.dropout(x)           
        x = F.relu(self.fc2(x))       
        x = self.dropout(x)           
        x = self.fc3(x)               
        return x
    
class LH_Direct(nn.Module):
    def __init__(self, in_dim, hid_dim, K, dprate, dropout, is_bns, act_fn, type='tri', pair_repr_dim=None):
        super(LH_Direct, self).__init__()
        self.encoder = ChebNetII(
            num_features=in_dim,
            hidden=hid_dim,
            K=K,
            dprate=dprate,
            dropout=dropout,
            is_bns=is_bns,
            act_fn=act_fn,
            pair_repr_dim=pair_repr_dim,
            node_dim=in_dim,
        )
        self.act_fn = nn.ReLU()
        self.alpha = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        self.beta = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        if type=='tri':
            self.mlp_input = 1489
        elif type=='pub':
            self.mlp_input = 881
        elif type=='maccs':
            self.mlp_input = 167
        elif type=='erg':
            self.mlp_input = 441
        self.mlp = MLP(input_size=self.mlp_input, hidden_size=1024, output_size=hid_dim, dropout=0.2)

    # Uni-Mol pair repr
    def _extract_pair_repr(self, data, edge_index, device):
        pair_repr_flat = None
        edge_index_for_pair = None

        if hasattr(data, 'pair_repr_edge') and data.pair_repr_edge is not None:
            pair_repr_edge = data.pair_repr_edge
            if torch.is_tensor(pair_repr_edge) and pair_repr_edge.numel() > 0:
                pair_repr_flat = pair_repr_edge.to(device=device, dtype=torch.float32)
                edge_index_for_pair = edge_index

        if pair_repr_flat is None and hasattr(data, 'pair_repr_full') and data.pair_repr_full is not None:
            pair_repr_full = data.pair_repr_full
            if (
                torch.is_tensor(pair_repr_full)
                and pair_repr_full.dim() == 3
                and pair_repr_full.size(0) == pair_repr_full.size(1)
            ):
                pair_repr_full = pair_repr_full.to(device=device, dtype=torch.float32)
                pair_repr_flat = pair_repr_full[edge_index[0], edge_index[1]]
                edge_index_for_pair = edge_index

        return pair_repr_flat, edge_index_for_pair

    def get_embedding(self, data, device): 
        
        feat, edge_index, batch, y ,edge_attr, fp, w= data.x, data.edge_index, data.batch, data.y, data.edge_attr, data.fps, data.w
        feat = feat.to(device)
        edge_index = edge_index.to(device)
        edge_attr = edge_attr.to(device)
        batch = batch.to(device)
        fp = fp.to(device)
        y = y.to(device)

        pair_repr_flat, edge_index_for_pair = self._extract_pair_repr(data, edge_index, device)
        high_pair_repr = pair_repr_flat.clone() if pair_repr_flat is not None else None
        low_pair_repr = pair_repr_flat.clone() if pair_repr_flat is not None else None

        h1, high_pair_repr = self.encoder(
            x=feat,
            edge_index=edge_index,
            highpass=True,
            pair_repr=high_pair_repr,
            edge_index_for_pair=edge_index_for_pair,
        )
        high_x_mean = gmp(h1, batch)
        
        h2, low_pair_repr = self.encoder(
            x=feat,
            edge_index=edge_index,
            highpass=False,
            pair_repr=low_pair_repr,
            edge_index_for_pair=edge_index_for_pair,
        )
        low_x_mean = gmp(h2, batch)
        
        h = torch.mul(self.alpha, h1) + torch.mul(self.beta, h2)
        spec_x_mean = gmp(h, batch)
        
        fp = fp.reshape(len(fp)//self.mlp_input, self.mlp_input)
        x_fp = self.mlp(fp)
        
        return low_x_mean, high_x_mean, spec_x_mean, x_fp, y


    def forward(self, data, device):
        # positive
        feat, edge_index, batch, y ,edge_attr, fp, w= data.x, data.edge_index, data.batch, data.y, data.edge_attr, data.fps, data.w
        feat = feat.to(device)
        edge_index = edge_index.to(device)
        edge_attr = edge_attr.to(device)
        batch = batch.to(device)
        fp = fp.to(device)
        
        pair_repr_flat, edge_index_for_pair = self._extract_pair_repr(data, edge_index, device)
        high_pair_repr = pair_repr_flat.clone() if pair_repr_flat is not None else None
        low_pair_repr = pair_repr_flat.clone() if pair_repr_flat is not None else None

        h1, high_pair_repr = self.encoder(
            x=feat,
            edge_index=edge_index,
            edge_weight=edge_attr,
            highpass=True,
            pair_repr=high_pair_repr,
            edge_index_for_pair=edge_index_for_pair,
        )
        high_x_mean = gmp(h1, batch)
        h2, low_pair_repr = self.encoder(
            x=feat,
            edge_index=edge_index,
            edge_weight=edge_attr,
            highpass=False,
            pair_repr=low_pair_repr,
            edge_index_for_pair=edge_index_for_pair,
        )
        #print("h2", h2.shape)#4375,512
        low_x_mean = gmp(h2, batch)
        #print("low_x_mean", low_x_mean.shape) #128,5102
        h = torch.mul(self.alpha, h1) + torch.mul(self.beta, h2)
        #print("h", h.shape) #
        spec_x_mean = gmp(h, batch)
        #print("sepc_x_mean", spec_x_mean.shape)
        
        fp = fp.reshape(len(fp)//self.mlp_input, self.mlp_input)
        x_fp = self.mlp(fp)
        
        
        return low_x_mean, high_x_mean, spec_x_mean, x_fp


class ChebNetII(torch.nn.Module):
    def __init__(self, num_features, hidden=512, K=10, dprate=0.50, dropout=0.50, is_bns=False, act_fn='relu',
                 pair_repr_dim=None, node_dim=None):
        super(ChebNetII, self).__init__()
        self.lin1 = Linear(num_features, hidden)

        self.prop1 = ChebnetII_prop(
            K=K,
            pair_repr_dim=pair_repr_dim,
            node_dim=num_features if node_dim is None else node_dim,
        )
        assert act_fn in ['relu', 'prelu']
        self.act_fn = nn.PReLU() if act_fn == 'prelu' else nn.ReLU()
        self.bn = torch.nn.BatchNorm1d(num_features, momentum=0.01)
        self.is_bns = is_bns
        self.dprate = dprate
        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        self.prop1.reset_parameters()
        self.lin1.reset_parameters()

    def forward(self, x, edge_index, edge_weight= None, highpass=True, pair_repr=None, edge_index_for_pair=None):

        if self.dprate == 0.0:
            x, pair_repr = self.prop1(
                x,
                edge_index,
                edge_weight=edge_weight,
                highpass=highpass,
                pair_repr=pair_repr,
                edge_index_for_pair=edge_index_for_pair,
            )
        else:
            x = F.dropout(x, p=self.dprate, training=self.training)
            x, pair_repr = self.prop1(
                x,
                edge_index,
                edge_weight=edge_weight,
                highpass=highpass,
                pair_repr=pair_repr,
                edge_index_for_pair=edge_index_for_pair,
            )


        x = F.dropout(x, p=self.dropout, training=self.training)

        if self.is_bns:
            x = self.bn(x)

        x = self.lin1(x)
        x = self.act_fn(x)

        return x, pair_repr
    
class GNNCon(torch.nn.Module):
    def __init__(self,eps=0. , train_eps=True, n_output=512, num_features_xt=78, output_dim=512, dropout=0.2, encoder1=None, encoder2=None):

        super(GNNCon, self).__init__()
        self.num_features_xt = num_features_xt
        self.n_output = n_output
        self.dropout = dropout
        self.output_dim = output_dim

        self.gat = encoder1
        self.gin = encoder2

        # predict head
        self.pre_head = nn.Sequential(
            nn.Linear(output_dim, 2048),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, self.n_output)
        )

    def forward(self, data1):
        x, edge_index, batch, x_size, edge_size, edge_attr = data1.x, data1.edge_index, data1.batch, data1.x_size, data1.edge_size, data1.edge_attr
        x = x.to('cuda')
        edge_index = edge_index.to('cuda')
        batch = batch.to('cuda')
        edge_attr = edge_attr.to('cuda')
        
        edge_index1, norm1 = get_laplacian(edge_index, edge_attr, normalization='sym', dtype=x.dtype)
                                          # num_nodes=x.size(self.node_dim))
        
        x1, weight = self.gat(x, edge_index, edge_attr, batch)
        out1 = self.pre_head(x1)

        x2 = self.gin(x, edge_index, edge_attr, batch)
        out2 = self.pre_head(x2)

        len_w = edge_index.shape[1]
        w1 = weight[1]
        ew1 = w1[:len_w]
        xw1 = w1[len_w:]

        return x1, out1, x2, out2, ew1, xw1
