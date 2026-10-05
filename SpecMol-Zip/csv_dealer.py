import pandas as pd

# 1. 读取原始 MUV 数据
input_path = "/home/zzy/SpecMol-code-original/dataset/MUV/raw/MUV.csv"
input_df = pd.read_csv(input_path, sep=",")

# 2. 提取 SMILES 和所有任务标签（保持原始顺序）
tasks = [
    "MUV-466", "MUV-548", "MUV-600", "MUV-644", "MUV-652", "MUV-689",
    "MUV-692", "MUV-712", "MUV-713", "MUV-733", "MUV-737", "MUV-810",
    "MUV-832", "MUV-846", "MUV-852", "MUV-858", "MUV-859"
]
smiles_list = input_df["smiles"]
labels = input_df[tasks]

# # 3. 将标签中的 NaN 替换为 0（保持原始处理逻辑）
# labels = labels.fillna(0)

# 4. 合并 SMILES 和标签，生成与原始文件格式一致的新 CSV
output_df = pd.concat([smiles_list, labels], axis=1)

# 5. 保存为新的 CSV 文件（无列名，逗号分隔）
output_path = "/home/zzy/SpecMol-code-original/dataset/MUV/raw/smiles.csv"
output_df.to_csv(output_path, index=False, header=False)

print(f"Saved {len(output_df)} lines to {output_path}")