from unimol_tools.data.datahub import DataHub
import os
import numpy as np
import pandas as pd

# 1. 路径（你现在在 SpecMol-Zip 目录下跑）
RAW_CSV = r"dataset/bace/raw/smiles.csv"          # 原始文件（无表头）
CSV_PATH = r"dataset/bace/raw/smiles_unimol.csv"  # 给 Uni-Mol 用的新文件

# 2. 把原始文件转成带表头的 csv（每次都重建，避免之前错的文件）
df_raw = pd.read_csv(RAW_CSV, header=None)
df_raw.columns = ["smiles", "label"]   # 第0列是SMILES，第1列是标签
df_raw.to_csv(CSV_PATH, index=False)

df = pd.read_csv(CSV_PATH)
print("转换后列名:", df.columns)
print("前3行:\n", df.head(3))

# 3. 用 Uni-Mol 的 DataHub 读这个新 csv
hub = DataHub(
    data=CSV_PATH,
    is_train=True,
    save_path="./unimol_out",
    task="classification",
    data_type="molecule",
    smiles_col="smiles",       # SMILES 列名
    target_cols=["label"],     # 标签列名
    # 划分配置：10 折 scaffold
    split="scaffold",
    kfold=10,
    split_seed=42,
    # 3D / unimolv2 配置
    model_name="unimolv2",
    conf_cache_level=2,
    sdf_save_path="./unimol_out/sdf",
)

# 4. 把划分 & 3D 特征存下来
out_dir = os.path.join(hub.save_path or ".", "splits")
os.makedirs(out_dir, exist_ok=True)

split_nfolds = hub.data["split_nfolds"]   # [(train_idx, test_idx), ...]
kfold = hub.kfold
method = hub.method
split_seed = hub.split_seed

# 用 fold_id 表示每个样本属于哪一折
n_samples = len(df)
fold_ids = np.zeros(n_samples, dtype=int)
for fold, (tr_idx, te_idx) in enumerate(split_nfolds):
    fold_ids[te_idx] = fold

# ✅ 只存 fold_ids，不再存 split_nfolds（避免你刚才那个错误）
np.save(
    os.path.join(out_dir, f"fold_ids_{method}_k{kfold}_seed{split_seed}.npy"),
    fold_ids,
)

# 保存带 fold_id 的 csv：smiles + label + fold_id
df["fold_id"] = fold_ids
csv_out = os.path.join(
    out_dir, f"data_with_folds_{method}_k{kfold}_seed{split_seed}.csv"
)
df.to_csv(csv_out, index=False)

# 保存 unimol 的 3D 输入特征（no_h_list），用 object 数组存
unimol_input_arr = np.array(hub.data["unimol_input"], dtype=object)
np.save(
    os.path.join(
        out_dir, f"unimol_input_{method}_k{kfold}_seed{split_seed}.npy"
    ),
    unimol_input_arr,
)

print("\n✅ 完成")
print("带 fold 的 CSV:", csv_out)
print("3D 特征文件:", f"unimol_input_{method}_k{kfold}_seed{split_seed}.npy")
