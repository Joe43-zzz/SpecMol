import torch

# ===== 把这两个路径改成你的旧 / 新 pt =====
OLD_PATH = r"C:\Users\zhoutianyang\Desktop\SpecMol-Zip\check\bace_train.pt"
NEW_PATH = r"C:\Users\zhoutianyang\Desktop\SpecMol-Zip\SpecMol-Zip\down_task\processed\bace_train.pt"  # 你刚重写的

def inspect_pt(path, tag):
    print(f"\n================ {tag} =================")
    data, slices = torch.load(path, map_location="cpu")

    # data.keys 可能是方法，也可能是属性，保险一点这样写：
    data_keys = data.keys() if callable(getattr(data, "keys", None)) else data.keys
    print("data keys   :", data_keys)
    print("slices keys :", list(slices.keys()))

    # 图的数量：用 y 的切片长度
    num_graphs = slices['y'].size(0) - 1
    print("图个数:", num_graphs)

    def show_graph(idx: int):
        print(f"\n------ {tag} | Graph {idx} ------")

        # 节点特征 x
        if hasattr(data, 'x') and 'x' in slices:
            x_start = slices['x'][idx].item()
            x_end   = slices['x'][idx + 1].item()
            x = data.x[x_start:x_end]
            print("x.shape:", x.shape)

        # 边索引 edge_index 维度是 [2, num_edges_total]，沿着第二维切
        if hasattr(data, 'edge_index') and 'edge_index' in slices:
            e_start = slices['edge_index'][idx].item()
            e_end   = slices['edge_index'][idx + 1].item()
            edge_index = data.edge_index[:, e_start:e_end]
            print("edge_index.shape:", edge_index.shape)

        # 边特征 edge_attr 维度是 [num_edges_total, dim]
        if hasattr(data, 'edge_attr') and data.edge_attr is not None and 'edge_attr' in slices:
            ea_start = slices['edge_attr'][idx].item()
            ea_end   = slices['edge_attr'][idx + 1].item()
            edge_attr = data.edge_attr[ea_start:ea_end]
            print("edge_attr.shape:", edge_attr.shape)

        # 3D 边权 edge_weight_3d（新 pt 才会有）
        if hasattr(data, 'edge_weight_3d') and 'edge_weight_3d' in slices:
            ew_start = slices['edge_weight_3d'][idx].item()
            ew_end   = slices['edge_weight_3d'][idx + 1].item()
            ew = data.edge_weight_3d[ew_start:ew_end]
            print("edge_weight_3d.shape:", ew.shape)

        # 指纹 fps（一般是图级向量）
        if hasattr(data, 'fps') and 'fps' in slices:
            fps_start = slices['fps'][idx].item()
            fps_end   = slices['fps'][idx + 1].item()
            fps = data.fps[fps_start:fps_end]
            print("fps.shape:", fps.shape)

        # 标签 y
        if hasattr(data, 'y') and 'y' in slices:
            y_start = slices['y'][idx].item()
            y_end   = slices['y'][idx + 1].item()
            y = data.y[y_start:y_end]
            print("y:", y.view(-1).tolist())

        # 样本权重 w
        if hasattr(data, 'w') and 'w' in slices:
            w_start = slices['w'][idx].item()
            w_end   = slices['w'][idx + 1].item()
            w = data.w[w_start:w_end]
            print("w:", w.view(-1).tolist())

    # 看前 3 个图
    for i in range(min(3, num_graphs)):
        show_graph(i)

    return data, slices


if __name__ == "__main__":
    old_data, old_slices = inspect_pt(OLD_PATH, "OLD")
    new_data, new_slices = inspect_pt(NEW_PATH, "NEW")

