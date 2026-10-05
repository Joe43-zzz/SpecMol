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
    def __init__(self, in_dim, hid_dim, K, dprate, dropout, is_bns, act_fn, type='tri'):
        super(LH_Direct, self).__init__()
        self.encoder = ChebNetII(
            num_features=in_dim,
            hidden=hid_dim,
            K=K,
            dprate=dprate,
            dropout=dropout,
            is_bns=is_bns,
            act_fn=act_fn,
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

    @staticmethod
    def _edge_weight(data, device):
        if hasattr(data, 'edge_weight_3d'):
            return data.edge_weight_3d.to(device)
        if hasattr(data, 'edge_attr') and data.edge_attr is not None and data.edge_attr.dim() == 1:
            return data.edge_attr.to(device)
        return None

    def get_embedding(self, data, device):

        feat, edge_index, batch, y, fp = data.x, data.edge_index, data.batch, data.y, data.fps
        feat = feat.to(device)
        edge_index = edge_index.to(device)
        edge_weight = self._edge_weight(data, device)
        batch = batch.to(device)
        fp = fp.to(device)
        y = y.to(device)

        h1 = self.encoder(x=feat, edge_index=edge_index, edge_weight=edge_weight, highpass=True)
        high_x_mean = gmp(h1, batch)

        h2 = self.encoder(x=feat, edge_index=edge_index, edge_weight=edge_weight, highpass=False)
        low_x_mean = gmp(h2, batch)

        h = torch.mul(self.alpha, h1) + torch.mul(self.beta, h2)
        spec_x_mean = gmp(h, batch)

        fp = fp.reshape(len(fp)//self.mlp_input, self.mlp_input)
        x_fp = self.mlp(fp)

        return low_x_mean, high_x_mean, spec_x_mean, x_fp, y


    def forward(self, data, device):
        feat, edge_index, batch, fp = data.x, data.edge_index, data.batch, data.fps
        feat = feat.to(device)
        edge_index = edge_index.to(device)
        edge_weight = self._edge_weight(data, device)
        batch = batch.to(device)
        fp = fp.to(device)

        h1 = self.encoder(x=feat, edge_index=edge_index, edge_weight=edge_weight, highpass=True)
        high_x_mean = gmp(h1, batch)
        h2 = self.encoder(x=feat, edge_index=edge_index, edge_weight=edge_weight, highpass=False)
        low_x_mean = gmp(h2, batch)
        h = torch.mul(self.alpha, h1) + torch.mul(self.beta, h2)
        spec_x_mean = gmp(h, batch)

        fp = fp.reshape(len(fp)//self.mlp_input, self.mlp_input)
        x_fp = self.mlp(fp)

        return low_x_mean, high_x_mean, spec_x_mean, x_fp


class ChebNetII(torch.nn.Module):
    def __init__(self, num_features, hidden=512, K=10, dprate=0.50, dropout=0.50, is_bns=False, act_fn='relu'):
        super(ChebNetII, self).__init__()
        self.lin1 = Linear(num_features, hidden)

        self.prop1 = ChebnetII_prop(K=K)
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

    def forward(self, x, edge_index, edge_weight=None, highpass=True):

        if self.dprate == 0.0:
            x = self.prop1(x, edge_index, edge_weight=edge_weight, highpass=highpass)
        else:
            x = F.dropout(x, p=self.dprate, training=self.training)
            x = self.prop1(x, edge_index, edge_weight=edge_weight, highpass=highpass)

        x = F.dropout(x, p=self.dropout, training=self.training)

        if self.is_bns:
            x = self.bn(x)

        x = self.lin1(x)
        x = self.act_fn(x)

        return x
