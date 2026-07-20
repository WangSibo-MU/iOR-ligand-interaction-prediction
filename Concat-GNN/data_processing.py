import numpy as np
from rdkit import Chem
import torch
from collections import defaultdict
from sklearn.model_selection import train_test_split
import os
import argparse
from torch_geometric.data import Data
import pickle

atom_dict = defaultdict(lambda: len(atom_dict))
fingerprint_dict = defaultdict(lambda: len(fingerprint_dict))
edge_dict = defaultdict(lambda: len(edge_dict))

# ========== 化合物分子图提取部分 ==========
def create_atoms(mol):
    atoms = []
    for a in mol.GetAtoms():
        atom_type = a.GetSymbol()  # 元素符号
        degree = a.GetDegree()  # 原子度
        formal_charge = a.GetFormalCharge()  # 形式电荷
        is_aromatic = a.GetIsAromatic()  # 是否芳香原子
        hybridization = a.GetHybridization()  # 杂化状态

        atom_features = (atom_type, degree, formal_charge, is_aromatic, hybridization)
        atoms.append(atom_features)

    atoms = [atom_dict[a] for a in atoms]
    return np.array(atoms)

def create_ijbonddict(mol):
    """创建邻接原子与键类型的映射字典"""
    i_jbond_dict = defaultdict(lambda: [])
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bond = str(b.GetBondType())
        i_jbond_dict[i].append((j, bond))
        i_jbond_dict[j].append((i, bond))
    return i_jbond_dict

def extract_fingerprints(atoms, i_jbond_dict, radius):
    """采用 Weisfeiler-Lehman 算法提取子图指纹特征"""
    if len(atoms) == 1 or radius == 0:
        fingerprints = [fingerprint_dict[a] for a in atoms]
    else:
        nodes = atoms
        i_jedge_dict = i_jbond_dict
        for _ in range(radius):
            fingerprints = []
            for i, j_edge in i_jedge_dict.items():
                neighbors = [(nodes[j], edge) for j, edge in j_edge]
                fingerprint = (nodes[i], tuple(sorted(neighbors)))
                fingerprints.append(fingerprint_dict[fingerprint])
            nodes = fingerprints
            _i_jedge_dict = defaultdict(lambda: [])
            for i, j_edge in i_jbond_dict.items():
                for j, edge in j_edge:
                    both_side = tuple(sorted((nodes[i], nodes[j])))
                    edge = edge_dict[(both_side, edge)]
                    _i_jedge_dict[i].append((j, edge))
            i_jbond_dict = _i_jedge_dict

    node_features = []
    for atom, fingerprint in zip(atoms, fingerprints):
        node_features.append([float(atom), float(fingerprint)])

    return np.array(node_features, dtype=np.float32)

def create_adjacency(mol, max_nodes=50):
    """生成分子图的邻接矩阵"""
    adjacency = Chem.GetAdjacencyMatrix(mol)
    if adjacency.shape[0] < max_nodes:
        adjacency = np.pad(adjacency, ((0, max_nodes - adjacency.shape[0]), (0, max_nodes - adjacency.shape[1])), mode='constant')
    else:
        adjacency = adjacency[:max_nodes, :max_nodes]
    return np.array(adjacency)

def smiles_to_graph(smiles, radius, max_nodes=50, max_edges=100):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"无效的 SMILES: {smiles}")
        return None

    atoms = create_atoms(mol)
    i_jbond_dict = create_ijbonddict(mol)
    fingerprints = extract_fingerprints(atoms, i_jbond_dict, radius)

    if fingerprints.shape[0] < max_nodes:
        pad_width = ((0, max_nodes - fingerprints.shape[0]), (0, 0))
        fingerprints = np.pad(fingerprints, pad_width, mode='constant')
    else:
        fingerprints = fingerprints[:max_nodes]

    edge_index = []
    edge_features = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_index.append((i, j))
        edge_index.append((j, i))

        bond_feature = [
            bond.GetBondTypeAsDouble(),
            bond.IsInRing(),
            bond.GetIsConjugated(),
            bond.GetStereo(),
        ]
        edge_features.append(bond_feature)
        edge_features.append(bond_feature)

    edge_index = np.array(edge_index, dtype=np.int64).T
    edge_features = np.array(edge_features, dtype=np.float32)

    if edge_index.shape[1] < max_edges:
        pad_width = ((0, 0), (0, max_edges - edge_index.shape[1]))
        edge_index = np.pad(edge_index, pad_width, mode='constant')
        edge_features = np.pad(edge_features, ((0, max_edges - edge_features.shape[0]), (0, 0)), mode='constant')
    else:
        edge_index = edge_index[:, :max_edges]
        edge_features = edge_features[:max_edges]

    x = torch.tensor(fingerprints, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_attr = torch.tensor(edge_features, dtype=torch.float)

    adjacency = create_adjacency(mol, max_nodes=max_nodes)
    adjacency = torch.tensor(adjacency, dtype=torch.float)

    x_with_adjacency = torch.cat([x, adjacency], dim=1)

    adjacency_edge = adjacency[edge_index[0], edge_index[1]].unsqueeze(1)
    edge_attr_with_adjacency = torch.cat([edge_attr, adjacency_edge], dim=1)

    data = Data(x=x_with_adjacency, edge_index=edge_index, edge_attr=edge_attr_with_adjacency)
    return data

# ========== 蛋白质氨基酸序列对齐 ==========
def process_protein_sequence(protein, max_len=480):
    if len(protein) > max_len:
        return protein[:max_len]
    else:
        return protein.ljust(max_len)


def print_label_distribution(labels, subset_name):
    """打印二分类标签分布。"""
    labels = np.asarray(labels)
    total = len(labels)
    pos = int(np.sum(labels == 1))
    neg = int(np.sum(labels == 0))
    pos_ratio = pos / total if total > 0 else 0.0
    neg_ratio = neg / total if total > 0 else 0.0
    print(
        f"  {subset_name}: {total} samples "
        f"(Pos: {pos}, Neg: {neg}, Pos ratio: {pos_ratio:.4f}, Neg ratio: {neg_ratio:.4f})"
    )


def _build_group_indices(groups):
    """建立 group -> 样本索引 的映射。"""
    group_to_indices = defaultdict(list)
    for idx, g in enumerate(groups):
        group_to_indices[str(g).strip()].append(idx)
    return group_to_indices


def _score_group_split(test_n, test_pos, total_n, total_pos, test_size,
                       size_weight=1.0, label_weight=2.0, ratio_weight=1.0):
    """
    给候选 group split 打分。分数越低越好。
    同时优化：test size、正负样本数、正样本比例，并惩罚单类别子集。
    """
    total_neg = total_n - total_pos
    test_neg = test_n - test_pos

    target_test_n = total_n * test_size
    target_test_pos = total_pos * test_size
    target_test_neg = total_neg * test_size

    size_error = abs(test_n - target_test_n) / max(total_n, 1)
    pos_error = abs(test_pos - target_test_pos) / max(total_pos, 1)
    neg_error = abs(test_neg - target_test_neg) / max(total_neg, 1)
    label_error = pos_error + neg_error

    global_pos_ratio = total_pos / total_n if total_n > 0 else 0.0
    test_pos_ratio = test_pos / test_n if test_n > 0 else 0.0
    ratio_error = abs(test_pos_ratio - global_pos_ratio)

    penalty = 0.0
    train_pos = total_pos - test_pos
    train_neg = total_neg - test_neg
    if total_pos > 0 and (test_pos == 0 or train_pos == 0):
        penalty += 1000.0
    if total_neg > 0 and (test_neg == 0 or train_neg == 0):
        penalty += 1000.0

    return size_weight * size_error + label_weight * label_error + ratio_weight * ratio_error + penalty


def stratified_group_cold_start_split(groups, labels, test_size=0.2, random_state=42,
                                      n_trials=5000, candidates_per_step=64,
                                      size_weight=1.0, label_weight=2.0,
                                      ratio_weight=1.0):
    """
    分组分层 cold-start 划分。

    作用：
    1. 同一个 ligand/protein group 只进入 train 或 test，避免 cold-start 泄漏；
    2. test size 尽量接近 test_size；
    3. train/test 正负样本比例尽量接近整体正负比例。

    返回：train_indices, test_indices, split_info
    """
    if not 0 < test_size < 1:
        raise ValueError(f"test_size must be between 0 and 1, got {test_size}")

    rng = np.random.default_rng(random_state)
    groups = np.asarray([str(g).strip() for g in groups], dtype=object)
    labels = np.asarray(labels)

    unique_labels = set(np.unique(labels).tolist())
    if not unique_labels.issubset({0, 1}):
        raise ValueError(f"Only binary labels 0/1 are supported, got {sorted(unique_labels)}")

    total_n = len(labels)
    total_pos = int(np.sum(labels == 1))
    total_neg = total_n - total_pos

    if total_n == 0:
        raise ValueError("Cannot split an empty dataset.")
    if total_pos == 0 or total_neg == 0:
        raise ValueError("Cannot perform label-stratified split because only one class exists.")

    group_to_indices = _build_group_indices(groups)
    unique_groups = np.array(list(group_to_indices.keys()), dtype=object)
    if len(unique_groups) < 2:
        raise ValueError("Cannot perform cold-start split with fewer than two unique groups.")

    group_n = {}
    group_pos = {}
    for g, idxs in group_to_indices.items():
        idxs_arr = np.asarray(idxs, dtype=int)
        group_n[g] = len(idxs_arr)
        group_pos[g] = int(np.sum(labels[idxs_arr] == 1))

    target_test_n = total_n * test_size
    best_score = np.inf
    best_test_groups = None
    best_stats = None

    for trial in range(max(1, int(n_trials))):
        remaining = unique_groups.copy()
        rng.shuffle(remaining)
        remaining = list(remaining)

        test_groups = []
        test_n = 0
        test_pos = 0

        while remaining and test_n < target_test_n:
            if len(remaining) <= candidates_per_step:
                candidate_positions = np.arange(len(remaining))
            else:
                candidate_positions = rng.choice(len(remaining), size=candidates_per_step, replace=False)

            best_candidate_pos = None
            best_candidate_score = np.inf
            for pos in candidate_positions:
                g = remaining[int(pos)]
                cand_n = test_n + group_n[g]
                cand_pos = test_pos + group_pos[g]
                cand_score = _score_group_split(
                    cand_n, cand_pos, total_n, total_pos, test_size,
                    size_weight=size_weight,
                    label_weight=label_weight,
                    ratio_weight=ratio_weight
                )
                if cand_score < best_candidate_score:
                    best_candidate_score = cand_score
                    best_candidate_pos = int(pos)

            selected_group = remaining.pop(best_candidate_pos)
            test_groups.append(selected_group)
            test_n += group_n[selected_group]
            test_pos += group_pos[selected_group]

        score = _score_group_split(
            test_n, test_pos, total_n, total_pos, test_size,
            size_weight=size_weight,
            label_weight=label_weight,
            ratio_weight=ratio_weight
        )

        if score < best_score:
            best_score = score
            best_test_groups = set(test_groups)
            best_stats = {
                'score': float(score),
                'trial': int(trial),
                'test_n': int(test_n),
                'test_pos': int(test_pos),
                'test_neg': int(test_n - test_pos),
                'test_group_count': int(len(test_groups)),
            }

    if best_test_groups is None:
        raise RuntimeError("Failed to construct a stratified group cold-start split.")

    train_indices = []
    test_indices = []
    for g, idxs in group_to_indices.items():
        if g in best_test_groups:
            test_indices.extend(idxs)
        else:
            train_indices.extend(idxs)

    train_indices = np.asarray(train_indices, dtype=int)
    test_indices = np.asarray(test_indices, dtype=int)
    rng.shuffle(train_indices)
    rng.shuffle(test_indices)

    train_groups = set(groups[train_indices].tolist())
    test_groups = set(groups[test_indices].tolist())
    overlap = train_groups.intersection(test_groups)
    if overlap:
        raise RuntimeError(f"Group leakage detected: {len(overlap)} groups appear in both train and test.")

    split_info = {
        **best_stats,
        'train_n': int(len(train_indices)),
        'train_pos': int(np.sum(labels[train_indices] == 1)),
        'train_neg': int(np.sum(labels[train_indices] == 0)),
        'train_group_count': int(len(train_groups)),
        'total_group_count': int(len(unique_groups)),
        'target_test_size': float(test_size),
        'actual_test_size': float(len(test_indices) / total_n),
        'global_pos_ratio': float(total_pos / total_n),
        'train_pos_ratio': float(np.sum(labels[train_indices] == 1) / max(len(train_indices), 1)),
        'test_pos_ratio': float(np.sum(labels[test_indices] == 1) / max(len(test_indices), 1)),
    }

    return train_indices, test_indices, split_info

# ========== 主处理流程 ==========
def process_compound_protein_data(file_path, radius=2, max_nodes=50, max_edges=100, 
                                 max_protein_len=480, test_size=0.2, random_state=42,
                                 ligand_cold_start=False, protein_cold_start=False, balance_samples=False,
                                 stratified_group_trials=5000):

    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    compounds = []
    proteins = []
    labels = []
    original_smiles = []
    invalid_count = 0
    
    for i, line in enumerate(lines):
        parts = line.strip().split()
        if len(parts) < 3:
            continue
            
        smiles = parts[0]
        protein_seq = parts[1]
        label = int(parts[2])
        
        compound_graph = smiles_to_graph(smiles, radius, max_nodes, max_edges)
        
        if compound_graph is None:
            invalid_count += 1
            print(f"跳过无效分子结构 行号 {i+1}: {smiles}")
            continue
        
        processed_protein = process_protein_sequence(protein_seq, max_protein_len)
        
        compounds.append(compound_graph)
        proteins.append(processed_protein)
        labels.append(label)
        original_smiles.append(smiles)
        
        if (i+1) % 1000 == 0:
            print(f"已处理 {i+1} 个样本...")
    
    print(f"\n原始数据解析完毕，有效样本数: {len(compounds)}")
    print(f"累计跳过无效分子数: {invalid_count}")
    
    # 注意：不要把 PyG Data 对象列表转换成 np.array。
    # torch_geometric.data.Data 是可迭代对象，np.array(..., dtype=object)
    # 可能把每个 Data 拆成 list，导致训练阶段 compound_graphs[0].x 报错。
    proteins_arr = np.array(proteins)
    labels_arr = np.array(labels)
    smiles_arr = np.array(original_smiles)

    # 选择划分策略
    if ligand_cold_start:
        print("\n[Split Mode] 分组分层配体冷启动 (Stratified Group Ligand Cold Start)...")
        train_indices, test_indices, split_info = stratified_group_cold_start_split(
            groups=smiles_arr,
            labels=labels_arr,
            test_size=test_size,
            random_state=random_state,
            n_trials=stratified_group_trials
        )
        print("  Stratified group split diagnostics:")
        print(f"    Total ligand groups: {split_info['total_group_count']}")
        print(f"    Train ligand groups: {split_info['train_group_count']}")
        print(f"    Test ligand groups: {split_info['test_group_count']}")
        print(f"    Target test size: {split_info['target_test_size']:.4f}")
        print(f"    Actual test size: {split_info['actual_test_size']:.4f}")
        print(f"    Global pos ratio: {split_info['global_pos_ratio']:.4f}")
        print(f"    Train pos ratio: {split_info['train_pos_ratio']:.4f}")
        print(f"    Test pos ratio: {split_info['test_pos_ratio']:.4f}")
        print(f"    Best split score: {split_info['score']:.6f} (trial {split_info['trial']})")

    elif protein_cold_start:
        print("\n[Split Mode] 分组分层蛋白冷启动 (Stratified Group Protein Cold Start)...")
        train_indices, test_indices, split_info = stratified_group_cold_start_split(
            groups=proteins_arr,
            labels=labels_arr,
            test_size=test_size,
            random_state=random_state,
            n_trials=stratified_group_trials
        )
        print("  Stratified group split diagnostics:")
        print(f"    Total protein groups: {split_info['total_group_count']}")
        print(f"    Train protein groups: {split_info['train_group_count']}")
        print(f"    Test protein groups: {split_info['test_group_count']}")
        print(f"    Target test size: {split_info['target_test_size']:.4f}")
        print(f"    Actual test size: {split_info['actual_test_size']:.4f}")
        print(f"    Global pos ratio: {split_info['global_pos_ratio']:.4f}")
        print(f"    Train pos ratio: {split_info['train_pos_ratio']:.4f}")
        print(f"    Test pos ratio: {split_info['test_pos_ratio']:.4f}")
        print(f"    Best split score: {split_info['score']:.6f} (trial {split_info['trial']})")

    else:
        print("\n[Split Mode] 随机分层划分 (Random Stratified)...")
        indices = np.arange(len(labels_arr))
        train_indices, test_indices = train_test_split(
            indices, test_size=test_size, random_state=random_state, stratify=labels_arr
        )

    compounds_train = [compounds[i] for i in train_indices]
    compounds_test = [compounds[i] for i in test_indices]
    proteins_train = proteins_arr[train_indices].tolist()
    proteins_test = proteins_arr[test_indices].tolist()
    labels_train = labels_arr[train_indices].tolist()
    labels_test = labels_arr[test_indices].tolist()
    smiles_train = smiles_arr[train_indices].tolist()
    smiles_test = smiles_arr[test_indices].tolist()

    print("\nInitial split label distribution:")
    print_label_distribution(labels_train, "Train before balancing")
    print_label_distribution(labels_test, "Test before balancing")

    # ================= 新增：切分后的集合内动态强制 1:1 平衡 =================
    if balance_samples:
        print("\n[Balance] 对划分后的训练集和测试集分别应用动态降采样平衡...")
        rng = np.random.default_rng(random_state)

        # 1. 训练集严格对等平衡
        labels_train_np = np.array(labels_train)
        pos_train_idx = np.where(labels_train_np == 1)[0]
        neg_train_idx = np.where(labels_train_np == 0)[0]
        train_min_count = min(len(pos_train_idx), len(neg_train_idx))
        
        pos_train_sampled = rng.choice(pos_train_idx, train_min_count, replace=False)
        neg_train_sampled = rng.choice(neg_train_idx, train_min_count, replace=False)
        balanced_train_idx = np.concatenate([pos_train_sampled, neg_train_sampled])
        rng.shuffle(balanced_train_idx)
        
        compounds_train = [compounds_train[i] for i in balanced_train_idx]
        proteins_train = [proteins_train[i] for i in balanced_train_idx]
        labels_train = [labels_train[i] for i in balanced_train_idx]
        smiles_train = [smiles_train[i] for i in balanced_train_idx]

        # 2. 测试集严格对等平衡
        labels_test_np = np.array(labels_test)
        pos_test_idx = np.where(labels_test_np == 1)[0]
        neg_test_idx = np.where(labels_test_np == 0)[0]
        test_min_count = min(len(pos_test_idx), len(neg_test_idx))
        
        pos_test_sampled = rng.choice(pos_test_idx, test_min_count, replace=False)
        neg_test_sampled = rng.choice(neg_test_idx, test_min_count, replace=False)
        balanced_test_idx = np.concatenate([pos_test_sampled, neg_test_sampled])
        rng.shuffle(balanced_test_idx)
        
        compounds_test = [compounds_test[i] for i in balanced_test_idx]
        proteins_test = [proteins_test[i] for i in balanced_test_idx]
        labels_test = [labels_test[i] for i in balanced_test_idx]
        smiles_test = [smiles_test[i] for i in balanced_test_idx]
    # =========================================================================

    if balance_samples:
        print("\nBalanced label distribution:")
        print_label_distribution(labels_train, "Train after balancing")
        print_label_distribution(labels_test, "Test after balancing")

    os.makedirs('processed_data', exist_ok=True)
    
    all_protein_chars = set()
    for p in proteins:
        all_protein_chars.update(p)
    all_protein_chars.add(' ')
    
    with open('processed_data/atom_dict.pkl', 'wb') as f:
        pickle.dump(dict(atom_dict), f)
    with open('processed_data/fingerprint_dict.pkl', 'wb') as f:
        pickle.dump(dict(fingerprint_dict), f)
    with open('processed_data/edge_dict.pkl', 'wb') as f:
        pickle.dump(dict(edge_dict), f)
    with open('processed_data/protein_chars.pkl', 'wb') as f:
        pickle.dump(sorted(all_protein_chars), f)
    
    print("\n特征词典与蛋白质字符集序列化完成。")
    
    # 物理持久化数据
    torch.save(compounds_train, 'processed_data/train_compounds.pt')
    torch.save(compounds_test, 'processed_data/test_compounds.pt')
    
    # 特别更新：在 npz 文件中增加 smiles 的保存，用作训练代码冷启动的交叉验证 Group 划分依据
    np.savez('processed_data/train_data.npz',
             proteins=np.array(proteins_train),
             labels=np.array(labels_train),
             smiles=np.array(smiles_train))
    
    np.savez('processed_data/test_data.npz',
             proteins=np.array(proteins_test),
             labels=np.array(labels_test),
             smiles=np.array(smiles_test))
    
    with open('processed_data/train.txt', 'w') as f_train:
        for s, p, l in zip(smiles_train, proteins_train, labels_train):
            f_train.write(f"{s}\t{p}\t{l}\n")
    
    with open('processed_data/test.txt', 'w') as f_test:
        for s, p, l in zip(smiles_test, proteins_test, labels_test):
            f_test.write(f"{s}\t{p}\t{l}\n")
    
    metadata = {
        'num_train': len(compounds_train),
        'num_test': len(compounds_test),
        'max_nodes': max_nodes,
        'max_edges': max_edges,
        'max_protein_len': max_protein_len,
        'atom_dict_size': len(atom_dict),
        'fingerprint_dict_size': len(fingerprint_dict),
        'edge_dict_size': len(edge_dict),
        'protein_chars_size': len(all_protein_chars)
    }
    
    with open('processed_data/metadata.txt', 'w') as f_meta:
        for key, value in metadata.items():
            f_meta.write(f"{key}: {value}\n")
    
    print("\n本地数据集保存成功:")
    print(f"- 训练集大小: {len(compounds_train)} (正样本: {np.sum(np.array(labels_train)==1)}, 负样本: {np.sum(np.array(labels_train)==0)})")
    print(f"- 测试集大小: {len(compounds_test)} (正样本: {np.sum(np.array(labels_test)==1)}, 负样本: {np.sum(np.array(labels_test)==0)})")
    
    return {
        'compounds_train': compounds_train,
        'compounds_test': compounds_test,
        'proteins_train': proteins_train,
        'proteins_test': proteins_test,
        'labels_train': labels_train,
        'labels_test': labels_test,
        'metadata': metadata
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process compound-protein interaction data.")
    parser.add_argument('--file_path', type=str, default='dataset.txt', help='Path to the dataset text file.')
    parser.add_argument('--ligand_cold_start', action='store_true', help='Enable ligand cold start (disjoint ligands across splits, default cold start mode).')
    parser.add_argument('--protein_cold_start', action='store_true', help='Enable protein cold start (disjoint proteins across splits).')
    parser.add_argument('--balance_samples', action='store_true', help='Force equal number of positive and negative samples via post-split downsampling.')
    parser.add_argument('--radius', type=int, default=2, help='Radius for Weisfeiler-Lehman algorithm.')
    parser.add_argument('--max_nodes', type=int, default=50, help='Maximum number of nodes.')
    parser.add_argument('--max_edges', type=int, default=100, help='Maximum number of edges.')
    parser.add_argument('--max_protein_len', type=int, default=480, help='Maximum truncated/padded length for proteins.')
    parser.add_argument('--test_size', type=float, default=0.2, help='Proportion of the dataset to include in the test split.')
    parser.add_argument('--random_state', type=int, default=42, help='Random state seed.')
    parser.add_argument('--stratified_group_trials', type=int, default=5000,
                        help='Number of randomized trials for stratified group cold-start split.')
    
    args = parser.parse_args()
    
    if args.ligand_cold_start and args.protein_cold_start:
        raise ValueError("Cannot enable both ligand cold start and protein cold start at the same time. Please choose one.")

    process_compound_protein_data(
        file_path=args.file_path,
        radius=args.radius,
        max_nodes=args.max_nodes,
        max_edges=args.max_edges,
        max_protein_len=args.max_protein_len,
        test_size=args.test_size,
        random_state=args.random_state,
        ligand_cold_start=args.ligand_cold_start,
        protein_cold_start=args.protein_cold_start,
        balance_samples=args.balance_samples,
        stratified_group_trials=args.stratified_group_trials
    )