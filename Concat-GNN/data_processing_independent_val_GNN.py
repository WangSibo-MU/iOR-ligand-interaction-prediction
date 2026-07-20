import numpy as np
from rdkit import Chem
import torch
from collections import defaultdict
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

# ========== 主处理流程 ==========
def process_compound_protein_data(file_path, radius=2, max_nodes=50, max_edges=100, 
                                 max_protein_len=480, random_state=42,
                                 balance_samples=False):

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

    # ================= 外部验证集内动态强制 1:1 平衡 =================
    if balance_samples:
        print("\n[Balance] 对独立外部验证集应用动态降采样平衡...")
        rng = np.random.default_rng(random_state)

        labels_np = np.array(labels)
        pos_idx = np.where(labels_np == 1)[0]
        neg_idx = np.where(labels_np == 0)[0]
        min_count = min(len(pos_idx), len(neg_idx))
        
        pos_sampled = rng.choice(pos_idx, min_count, replace=False)
        neg_sampled = rng.choice(neg_idx, min_count, replace=False)
        balanced_idx = np.concatenate([pos_sampled, neg_sampled])
        rng.shuffle(balanced_idx)
        
        compounds = [compounds[i] for i in balanced_idx]
        proteins_arr = proteins_arr[balanced_idx]
        labels_arr = labels_arr[balanced_idx]
        smiles_arr = smiles_arr[balanced_idx]
    # ==================================================================

    os.makedirs('processed_data', exist_ok=True)
    
    all_protein_chars = set()
    for p in proteins_arr:
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
    
    # 物理持久化独立外部验证集数据
    torch.save(compounds, 'processed_data/external_validation_compounds.pt')
    
    np.savez('processed_data/external_validation_data.npz',
             proteins=np.array(proteins_arr),
             labels=np.array(labels_arr),
             smiles=np.array(smiles_arr))
    
    with open('processed_data/external_validation.txt', 'w') as f_ext:
        for s, p, l in zip(smiles_arr, proteins_arr, labels_arr):
            f_ext.write(f"{s}\t{p}\t{l}\n")
    
    metadata = {
        'num_external_validation': len(compounds),
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
    
    print("\n独立外部验证集保存成功:")
    print(f"- 外部验证集大小: {len(compounds)} (正样本: {np.sum(labels_arr==1)}, 负样本: {np.sum(labels_arr==0)})")
    
    return {
        'compounds_external_validation': compounds,
        'proteins_external_validation': proteins_arr.tolist(),
        'labels_external_validation': labels_arr.tolist(),
        'smiles_external_validation': smiles_arr.tolist(),
        'metadata': metadata
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process compound-protein interaction data for independent external validation.")
    parser.add_argument('--file_path', type=str, default='dataset.txt', help='Path to the dataset text file.')
    parser.add_argument('--balance_samples', action='store_true', help='Force equal number of positive and negative samples via downsampling.')
    parser.add_argument('--radius', type=int, default=2, help='Radius for Weisfeiler-Lehman algorithm.')
    parser.add_argument('--max_nodes', type=int, default=50, help='Maximum number of nodes.')
    parser.add_argument('--max_edges', type=int, default=100, help='Maximum number of edges.')
    parser.add_argument('--max_protein_len', type=int, default=480, help='Maximum truncated/padded length for proteins.')
    parser.add_argument('--random_state', type=int, default=42, help='Random state seed.')
    
    args = parser.parse_args()

    process_compound_protein_data(
        file_path=args.file_path,
        radius=args.radius,
        max_nodes=args.max_nodes,
        max_edges=args.max_edges,
        max_protein_len=args.max_protein_len,
        random_state=args.random_state,
        balance_samples=args.balance_samples
    )