import os
from torch_geometric.data import InMemoryDataset
import torch
from create_data_DC import smile_to_graph
import deepchem as dc
import numpy as np
from rdkit.Chem import AllChem, MACCSkeys
from deepchem.feat.base_classes import MolecularFeaturizer
from pubchemfp import GetPubChemFPs
import pandas as pd
np.set_printoptions(threshold=np.inf)
from rdkit import Chem
from rdkit.Chem import rdchem
from rdkit.Chem import rdmolfiles, rdmolops
from deepchem.splits import ScaffoldSplitter
from deepchem.feat import CircularFingerprint
from deepchem.trans import BalancingTransformer
from torch_geometric.data import Data


class CombinedFingerprintsFeaturizer(MolecularFeaturizer):
    def __init__(self):
        super(CombinedFingerprintsFeaturizer,self).__init__()
        
    def _featurize(self,mol):
        fp = []
        
        pubchem_fp = GetPubChemFPs(mol)
        maccs_fp = AllChem.GetMACCSKeysFingerprint(mol)
        erg_fp = AllChem.GetErGFingerprint(mol,fuzzIncrement=0.3,maxPath=21,minPath=1)
        
        fp = np.concatenate([pubchem_fp, maccs_fp, erg_fp])
        
        return fp

    
class TestbedDataset(InMemoryDataset):
    def __init__(self, root='tmp', dataset='train', task='bbbp', type='tri', seed=9,
                 transform=None, pre_transform=None):
        
        super(TestbedDataset, self).__init__(root, transform, pre_transform)
        self.dataset = dataset
        self.task = task
        self.type = type
        self.seed = seed
        print('Processing data for task {}, dataset {}...'.format(task, dataset))
        # self.process(root, task)

        # # 加载对应数据集
        # if dataset == 'train':
        #     self.data, self.slices = torch.load(self.processed_paths[0])
        # elif dataset == 'valid':
        #     self.data, self.slices = torch.load(self.processed_paths[1])
        # elif dataset == 'test':
        #     self.data, self.slices = torch.load(self.processed_paths[2])
        # elif dataset == 'all':  # 新增'all'分支
        #     self.data, self.slices = torch.load(self.processed_paths[3])
        # 第一阶段：确保基础分片存在
        if not self._check_base_files_exist():
            print(f'Processing base splits for task {task}...')
            self._process_base_splits(root, task)

        # 第二阶段：处理特殊'all'数据集
        old_all = self._old_style_paths()[3]
        if dataset == 'all' and not os.path.exists(self.processed_paths[3]) and not os.path.exists(old_all):
            print(f'Processing full dataset for task {task}...')
            self._process_full_dataset(root, task)

        # 加载数据（优先使用带type/seed的新文件名，回退到老文件名）
        idx = ['train', 'valid', 'test', 'all'].index(dataset)
        target_path = self.processed_paths[idx]
        if not os.path.exists(target_path):
            target_path = self._old_style_paths()[idx]
        self.data, self.slices = torch.load(target_path)

    @property
    def processed_dir(self):
        # Check new-style (type/seed-tagged) names first, then old-style names
        new_style_files = [os.path.join(self.root, name) for name in self.processed_file_names]
        old_style_files = [os.path.join(self.root, f'{self.task}_{split}.pt')
                           for split in ('train', 'valid', 'test', 'all')]
        if any(os.path.exists(p) for p in new_style_files + old_style_files):
            return self.root
        return super().processed_dir

    @property
    def processed_file_names(self):
        # Include type and seed so different configurations don't share the same cache
        tag = f'{self.task}_{self.type}_{self.seed}'
        return [
            f'{tag}_train.pt',
            f'{tag}_valid.pt',
            f'{tag}_test.pt',
            f'{tag}_all.pt',
        ]

    def _old_style_paths(self):
        """Paths used by pre-built Uni-Mol .pt files (no type/seed suffix)."""
        base = self.processed_dir
        return [os.path.join(base, f'{self.task}_{split}.pt')
                for split in ('train', 'valid', 'test', 'all')]

    def _check_base_files_exist(self):
        """检查train/valid/test分片是否已存在 (new-style or old-style)"""
        new_style = all(os.path.exists(p) for p in self.processed_paths[:3])
        if new_style:
            return True
        old_paths = self._old_style_paths()
        return all(os.path.exists(p) for p in old_paths[:3])

    def _process_base_splits(self, root, task):
        """处理并保存基础分片（train/valid/test）"""
        dataset = self._load_raw_dataset(task)
        
        # 数据集分割
        splitter = ScaffoldSplitter()
        train, valid, test = splitter.train_valid_test_split(dataset)
        
        # 数据平衡处理
        transformer = BalancingTransformer(dataset=train)
        train = transformer.transform(train)
        valid = transformer.transform(valid)
        test = transformer.transform(test)
        
        # 保存分片
        self._save_dataset(train, 0)
        self._save_dataset(valid, 1)
        self._save_dataset(test, 2)

    def _process_full_dataset(self, root, task):
        """处理并保存完整数据集"""
        full_dataset = self._load_raw_dataset(task)
        self._save_dataset(full_dataset, 3)

    def _load_raw_dataset(self, task):
        """从原始文件加载完整数据"""
        csv_file = f'dataset/{task}/raw/smiles.csv'
        data = pd.read_csv(csv_file, header=None)

        smiles_all = data.iloc[:, 0].values
        properties_all = data.iloc[:, 1:].values
        smiles_all = [str(smile) for smile in smiles_all]

        # 过滤无效分子
        smiles, properties = [], []
        for i, smile in enumerate(smiles_all):
            mol = Chem.MolFromSmiles(smile)
            if mol is not None:  # 跳过无效分子
                smiles.append(smile)
                properties.append(properties_all[i])

        smiles = np.array(smiles)
        properties = np.array(properties)
        
        # 随机打乱（如果任务需要）
        if self.task != 'bbbp':
            np.random.seed(self.seed)
            indices = np.arange(len(smiles))
            np.random.shuffle(indices)
            smiles = smiles[indices]
            properties = properties[indices]

        # 特征提取
        comfp_featurizer = CombinedFingerprintsFeaturizer()
        features = comfp_featurizer.featurize(smiles)

        # 创建数据集
        n_samples = len(smiles)
        n_tasks = properties.shape[1]
        w = np.ones((n_samples, n_tasks))
        
        return dc.data.NumpyDataset(X=features, y=properties, w=w, ids=smiles)

    @staticmethod
    def _generate_3d_features(smile, n_atoms_expected):
        """
        Use RDKit to generate 3D coordinates and atom types for a SMILES string.

        Returns:
            pos:       [N, 3] float tensor of heavy-atom 3D coordinates
            atom_type: [N]    long  tensor of atomic numbers clamped to [0, 63]

            Returns (None, None) on any failure.
        """
        try:
            mol = Chem.MolFromSmiles(smile)
            if mol is None:
                return None, None

            mol_h = Chem.AddHs(mol)

            result = AllChem.EmbedMolecule(mol_h, randomSeed=42)
            if result == -1:
                result = AllChem.EmbedMolecule(mol_h, useRandomCoords=True, randomSeed=42)
                if result == -1:
                    return None, None

            try:
                AllChem.MMFFOptimizeMolecule(mol_h, maxIters=200)
            except Exception:
                pass  # optimisation failure is non-fatal

            mol_noH = Chem.RemoveHs(mol_h)
            n_atoms = mol_noH.GetNumAtoms()

            if n_atoms != n_atoms_expected:
                return None, None

            conf = mol_noH.GetConformer()
            pos = torch.zeros(n_atoms, 3)
            for idx in range(n_atoms):
                p = conf.GetAtomPosition(idx)
                pos[idx, 0] = p.x
                pos[idx, 1] = p.y
                pos[idx, 2] = p.z

            atom_type = torch.zeros(n_atoms, dtype=torch.long)
            for idx in range(n_atoms):
                atomic_num = mol_noH.GetAtomWithIdx(idx).GetAtomicNum()
                atom_type[idx] = min(atomic_num, 63)

            return pos, atom_type

        except Exception:
            return None, None

    def _save_dataset(self, dataset, path_idx):
        """通用保存方法"""
        data_list = []
        label_new = np.nan_to_num(dataset.y, nan=999)
        n_skipped_3d = 0

        for i in range(len(dataset)):
            try:
                smile = dataset.ids[i]
                label = label_new[i]
                weights = dataset.w[i]
                mfp = dataset.X[i]

                # 分子到图的转换
                x_size, features, edge_index, edge_features, atoms = smile_to_graph(smile)

                # === 生成3D特征（必须成功，否则跳过该分子）===
                pos, atom_type = self._generate_3d_features(smile, n_atoms_expected=x_size)
                if pos is None:
                    n_skipped_3d += 1
                    print(f"跳过分子 {i} (SMILES: {smile[:40]}...): 3D坐标生成失败")
                    continue

                # 处理空边的情况
                edge_index = torch.LongTensor(edge_index)
                if edge_index.dim() == 2 and edge_index.size(0) == 2:
                    pass  # 已经是正确形状
                else:
                    edge_index = edge_index.t().contiguous() if edge_index.numel() > 0 else edge_index

                # 构造Data对象
                GCNData = Data(
                    x=torch.Tensor(features) if features is not None else torch.empty(0),
                    edge_index=edge_index,
                    y=torch.Tensor(label),
                    edge_attr=torch.Tensor(edge_features) if edge_features is not None else torch.empty(0),
                    w=torch.Tensor(weights),
                    fps=torch.Tensor(mfp),
                    pos=pos,
                    atom_type=atom_type,
                )
                data_list.append(GCNData)
            except Exception as e:
                print(f"跳过无效样本 {i}: {str(e)}")
                continue

        if n_skipped_3d > 0:
            print(f"3D生成失败跳过: {n_skipped_3d} / {len(dataset)} 分子")

        # 应用预处理
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        # 确保数据有效后保存
        if len(data_list) > 0:
            data, slices = self.collate(data_list)
            torch.save((data, slices), self.processed_paths[path_idx])
            print(f'Saved {len(data_list)} graphs to {self.processed_paths[path_idx]}')
        else:
            raise RuntimeError("所有数据样本均无效，请检查数据源！")


def save_AUCs(AUCs, filename):
    with open(filename, 'a') as f:
        f.write('\t'.join(map(str, AUCs)) + '\n')

def save_RMSEs(RMSEs, filename):
    with open(filename, 'a') as f:
        f.write('\t'.join(map(str, RMSEs)) + '\n')


if __name__ == '__main__':
    test_smiles = [
        ('CCO',                              'ethanol'),
        ('c1ccccc1',                         'benzene'),
        ('CC(=O)OC1=CC=CC=C1C(=O)O',        'aspirin'),
    ]

    for smile, name in test_smiles:
        mol = Chem.MolFromSmiles(smile)
        n_atoms = mol.GetNumAtoms()
        pos, atom_type = TestbedDataset._generate_3d_features(smile, n_atoms)

        assert pos is not None,                    f'3D generation failed for {name}'
        assert pos.shape    == (n_atoms, 3),       f'Wrong pos shape for {name}: {pos.shape}'
        assert atom_type.shape == (n_atoms,),      f'Wrong atom_type shape for {name}'
        assert atom_type.dtype == torch.long,      f'atom_type dtype should be long for {name}'
        assert (atom_type >= 0).all() and (atom_type <= 63).all(), \
            f'atom_type out of range for {name}: {atom_type}'
        # diagonal of distance matrix should be 0
        dist_diag = torch.cdist(pos, pos, p=2).diagonal()
        assert torch.allclose(dist_diag, torch.zeros(n_atoms), atol=1e-5), \
            f'Non-zero diagonal distances for {name}'

        print(f'{name} ({smile}): pos {tuple(pos.shape)}, atom_type {atom_type.tolist()}')

    print('All 3D generation tests passed')
