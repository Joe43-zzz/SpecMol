import pandas as pd
import numpy as np

CSV = r"unimol_out/splits/data_with_folds_scaffold_k5_seed42.csv"
FOLD = r"unimol_out/splits/fold_ids_scaffold_k5_seed42.npy"
UNI  = r"unimol_out/splits/unimol_input_scaffold_k5_seed42.npy"

df = pd.read_csv(CSV)
fold_ids = np.load(FOLD)
uni = np.load(UNI, allow_pickle=True)

print("行数:", len(df))
print("fold_ids 长度:", len(fold_ids))
print("unimol_input 长度:", len(uni))
print("前3行：")
print(df.head(3))
print("前10个 fold_id:", fold_ids[:10])
