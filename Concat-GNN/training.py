import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
# 引入 StratifiedKFold 与 StratifiedGroupKFold 实现冷启动下的均衡交叉验证
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, recall_score, matthews_corrcoef, roc_curve
import warnings
import os
import argparse
import pickle
from utils import Config, CharTokenizer, CompoundGNN, ProteinTransformer, CPIPredictor

warnings.filterwarnings('ignore')
os.environ["LOKY_MAX_CPU_COUNT"] = "4" 
torch.manual_seed(42)
np.random.seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = Config()


def normalize_compound_graphs(compound_graphs):
    """Validate/repair loaded compound graphs.

    Expected format is list[torch_geometric.data.Data]. Some older preprocessed
    files may contain list-of-(key, value) pairs because PyG Data objects were
    converted through numpy before saving.
    """
    if len(compound_graphs) == 0:
        raise ValueError("train_compounds.pt is empty.")

    first = compound_graphs[0]
    if hasattr(first, 'x') and hasattr(first, 'edge_index'):
        return compound_graphs

    repaired = []
    for graph in compound_graphs:
        if hasattr(graph, 'x') and hasattr(graph, 'edge_index'):
            repaired.append(graph)
            continue

        # Possible legacy format: [('x', tensor), ('edge_index', tensor), ('edge_attr', tensor)]
        if isinstance(graph, (list, tuple)):
            try:
                graph_dict = dict(graph)
                if 'x' in graph_dict and 'edge_index' in graph_dict:
                    repaired.append(Data(**graph_dict))
                    continue
            except Exception:
                pass

        raise TypeError(
            "train_compounds.pt has invalid graph format. Expected each item to be "
            "torch_geometric.data.Data with attributes .x and .edge_index, but got "
            f"{type(graph).__name__}. Regenerate processed_data using the fixed data_processing.py."
        )

    print("Warning: repaired legacy list-form compound graphs into PyG Data objects. "
          "Regenerating processed_data with fixed data_processing.py is still recommended.")
    return repaired

class ORLigandDataset(Dataset):
    def __init__(self, compound_graphs, proteins, labels, protein_tokenizer, protein_max_len):
        self.compound_graphs = compound_graphs
        self.proteins = proteins
        self.labels = labels
        self.protein_tokenizer = protein_tokenizer
        self.protein_max_len = protein_max_len
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        graph = self.compound_graphs[idx]
        protein_seq = self.proteins[idx]
        protein_encoded = self.protein_tokenizer.encode(protein_seq, self.protein_max_len)
        label = self.labels[idx]
        return graph, protein_encoded, label

def collate_fn(batch):
    graphs = [item[0] for item in batch]
    proteins = torch.tensor([item[1] for item in batch], dtype=torch.long)
    labels = torch.tensor([item[2] for item in batch], dtype=torch.float)
    
    batch_graph = Batch.from_data_list(graphs)
    
    return batch_graph, proteins, labels

def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    
    all_labels = []
    for _, _, labels in train_loader:
        all_labels.extend(labels.numpy())
    all_labels = np.array(all_labels)
        
    for batch_idx, (compound_data, protein, labels) in enumerate(train_loader):
        compound_data = compound_data.to(device)
        protein = protein.to(device)
        labels = labels.float().to(device)
        
        optimizer.zero_grad()
        
        outputs = model(compound_data, protein)
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    return avg_loss

def evaluate_model(model, data_loader, device):
    model.eval()
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for compound_data, protein, labels in data_loader:
            compound_data = compound_data.to(device)
            protein = protein.to(device)
            labels = labels.float().to(device)
            outputs = model(compound_data, protein)
            if outputs.dim() == 0:
                batch_preds = [outputs.item()]
            else:
                batch_preds = outputs.cpu().numpy().tolist()
            predictions.extend(batch_preds)
            true_labels.extend(labels.cpu().numpy())
    
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)
    
    auc = roc_auc_score(true_labels, predictions) if len(np.unique(true_labels)) > 1 else 0.5
    
    optimal_threshold = 0.5
    
    binary_preds = (predictions > optimal_threshold).astype(int)
    
    acc = accuracy_score(true_labels, binary_preds)
    f1 = f1_score(true_labels, binary_preds) if len(np.unique(true_labels)) > 1 else 0
    recall = recall_score(true_labels, binary_preds) if np.sum(true_labels) > 0 else 0
    mcc = matthews_corrcoef(true_labels, binary_preds) if len(np.unique(true_labels)) > 1 else 0
    
    return auc, acc, f1, recall, mcc

def main():
    # ================= 命令行参数解析 =================
    parser = argparse.ArgumentParser(description="Train CPI Predictor with CV options.")
    parser.add_argument('--ligand_cold_start', action='store_true', help='Enable ligand cold start for cross-validation (SMILES group split).')
    parser.add_argument('--protein_cold_start', action='store_true', help='Enable protein cold start for cross-validation (Protein sequence group split).')
    parser.add_argument('--balance_samples', action='store_true', help='Force equal number of positive and negative samples via downsampling within each CV fold.')
    args = parser.parse_args()

    # 互斥安全检查（双冷启动不支持）
    if args.ligand_cold_start and args.protein_cold_start:
        raise ValueError("Cannot enable both ligand cold start and protein cold start at the same time. Please choose one.")

    print("Loading processed data...")
    try:
        compound_graphs = torch.load('processed_data/train_compounds.pt')
        compound_graphs = normalize_compound_graphs(compound_graphs)
        
        train_data = np.load('processed_data/train_data.npz')
        proteins = train_data['proteins']
        labels = train_data['labels']
        
        # 获取用于冷启动 Group 划分的 SMILES (需要确保在 data_processing 阶段保存了 smiles)
        smiles = train_data['smiles']
        
        print(f"Loaded {len(compound_graphs)} training compounds")
        print(f"Node feature dimension: {compound_graphs[0].x.shape[1]}")
    except Exception as e:
        print(f"Error loading processed data: {e}")
        return
        
    print("Loading protein character set...")
    try:
        with open('processed_data/protein_chars.pkl', 'rb') as f:
            protein_chars = pickle.load(f)
        protein_tokenizer = CharTokenizer(protein_chars)
        print(f"Loaded protein tokenizer with vocab size: {protein_tokenizer.vocab_size}")
    except Exception as e:
        print(f"Error loading protein chars: {e}")
        return
    
    node_in_dim = compound_graphs[0].x.shape[1]
    
    # ================= 交叉验证分发策略 =================
    if args.ligand_cold_start:
        print("\n[Cold Start Option] Using StratifiedGroupKFold (Ligand Cold Start) for Cross-Validation.")
        groups = np.array([s.strip() for s in smiles])
        kfold = StratifiedGroupKFold(n_splits=config.k_folds, shuffle=True, random_state=42)
        split_iterator = kfold.split(compound_graphs, labels, groups)
    elif args.protein_cold_start:
        print("\n[Cold Start Option] Using StratifiedGroupKFold (Protein Cold Start) for Cross-Validation.")
        groups = np.array([p.strip() for p in proteins])
        kfold = StratifiedGroupKFold(n_splits=config.k_folds, shuffle=True, random_state=42)
        split_iterator = kfold.split(compound_graphs, labels, groups)
    else:
        print("\n[Default Mode] Using Random Stratified (StratifiedKFold) for Cross-Validation.")
        kfold = StratifiedKFold(n_splits=config.k_folds, shuffle=True, random_state=42)
        split_iterator = kfold.split(compound_graphs, labels)

    results = []
    
    for fold, (train_idx, val_idx) in enumerate(split_iterator):
        print(f'\nFold {fold + 1}/{config.k_folds}')
        
        # 提取当前 Fold 的原始未均衡数据
        fold_train_compounds = [compound_graphs[i] for i in train_idx]
        fold_train_proteins = proteins[train_idx]
        fold_train_labels = labels[train_idx]
        
        fold_val_compounds = [compound_graphs[i] for i in val_idx]
        fold_val_proteins = proteins[val_idx]
        fold_val_labels = labels[val_idx]
        
        # ================= 折内动态平衡机制 =================
        if args.balance_samples:
            # 1. 训练子集严格下采样 1:1
            pos_train_idx = np.where(fold_train_labels == 1)[0]
            neg_train_idx = np.where(fold_train_labels == 0)[0]
            train_min_count = min(len(pos_train_idx), len(neg_train_idx))
            
            rng = np.random.default_rng(42 + fold)
            pos_train_sampled = rng.choice(pos_train_idx, train_min_count, replace=False)
            neg_train_sampled = rng.choice(neg_train_idx, train_min_count, replace=False)
            balanced_train_idx = np.concatenate([pos_train_sampled, neg_train_sampled])
            rng.shuffle(balanced_train_idx)
            
            fold_train_compounds = [fold_train_compounds[i] for i in balanced_train_idx]
            fold_train_proteins = fold_train_proteins[balanced_train_idx]
            fold_train_labels = fold_train_labels[balanced_train_idx]

            # 2. 验证子集严格下采样 1:1
            pos_val_idx = np.where(fold_val_labels == 1)[0]
            neg_val_idx = np.where(fold_val_labels == 0)[0]
            val_min_count = min(len(pos_val_idx), len(neg_val_idx))
            
            pos_val_sampled = rng.choice(pos_val_idx, val_min_count, replace=False)
            neg_val_sampled = rng.choice(neg_val_idx, val_min_count, replace=False)
            balanced_val_idx = np.concatenate([pos_val_sampled, neg_val_sampled])
            rng.shuffle(balanced_val_idx)
            
            fold_val_compounds = [fold_val_compounds[i] for i in balanced_val_idx]
            fold_val_proteins = fold_val_proteins[balanced_val_idx]
            fold_val_labels = fold_val_labels[balanced_val_idx]

        # 打印各 Fold 真实参与训练和验证的样本分布
        print(f"  - [Final Balanced] Train Set Size: {len(fold_train_labels)} (Pos: {int(np.sum(fold_train_labels==1))}, Neg: {int(np.sum(fold_train_labels==0))})")
        print(f"  - [Final Balanced] Val Set Size:   {len(fold_val_labels)} (Pos: {int(np.sum(fold_val_labels==1))}, Neg: {int(np.sum(fold_val_labels==0))})")
        
        train_dataset = ORLigandDataset(
            fold_train_compounds, fold_train_proteins, fold_train_labels,
            protein_tokenizer, config.protein_max_len
        )
        
        val_dataset = ORLigandDataset(
            fold_val_compounds, fold_val_proteins, fold_val_labels,
            protein_tokenizer, config.protein_max_len
        )
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=config.batch_size, 
            shuffle=True,
            collate_fn=collate_fn
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=config.batch_size, 
            shuffle=False,
            collate_fn=collate_fn
        )
        
        model = CPIPredictor(
            config, 
            protein_vocab_size=protein_tokenizer.vocab_size,
            node_in_dim=node_in_dim
        ).to(device)
        
        criterion = nn.BCELoss()
        optimizer = optim.Adam(
            model.parameters(), 
            lr=config.learning_rate, 
            weight_decay=config.weight_decay
        )
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=3, verbose=True
        )
        
        best_auc = 0
        best_model_state = None
        best_acc, best_f1, best_recall, best_mcc = 0, 0, 0, 0
        
        for epoch in range(config.num_epochs):
            train_loss = train_model(model, train_loader, criterion, optimizer, device)
            auc, acc, f1, recall, mcc = evaluate_model(model, val_loader, device)
            
            scheduler.step(auc)
            
            if (epoch + 1) % 5 == 0:
                print(f'Epoch {epoch + 1}/{config.num_epochs}, Loss: {train_loss:.4f}, '
                      f'Val AUC: {auc:.4f}, Acc: {acc:.4f}, F1: {f1:.4f}, Recall: {recall:.4f}, MCC: {mcc:.4f}')
            
            if auc > best_auc:
                best_auc = auc
                best_acc = acc
                best_f1 = f1
                best_recall = recall
                best_mcc = mcc
                best_model_state = model.state_dict()    
                
        print(f'Fold {fold + 1} Best Results - AUC: {best_auc:.4f}, Acc: {best_acc:.4f}, '
              f'F1: {best_f1:.4f}, Recall: {best_recall:.4f}, MCC: {best_mcc:.4f}')
        
        results.append({
            'auc': best_auc,
            'acc': best_acc,
            'f1': best_f1,
            'recall': best_recall,
            'mcc': best_mcc
        })
        
        del model
        torch.cuda.empty_cache()
    
    avg_auc = np.mean([r['auc'] for r in results])
    avg_acc = np.mean([r['acc'] for r in results])
    avg_f1 = np.mean([r['f1'] for r in results])
    avg_recall = np.mean([r['recall'] for r in results])
    avg_mcc = np.mean([r['mcc'] for r in results])
    
    print('\n5-Fold Cross Validation Average Results:')
    print(f'AUC: {avg_auc:.4f}')
    print(f'Accuracy: {avg_acc:.4f}')
    print(f'F1 Score: {avg_f1:.4f}')
    print(f'Recall: {avg_recall:.4f}')
    print(f'MCC: {avg_mcc:.4f}')
    
    print("\nTraining final model on entire dataset...")
    
    # 最终模型训练的数据平衡
    final_compounds = compound_graphs
    final_proteins = proteins
    final_labels = labels
    
    if args.balance_samples:
        pos_idx = np.where(labels == 1)[0]
        neg_idx = np.where(labels == 0)[0]
        min_count = min(len(pos_idx), len(neg_idx))
        
        rng = np.random.default_rng(42)
        pos_sampled = rng.choice(pos_idx, min_count, replace=False)
        neg_sampled = rng.choice(neg_idx, min_count, replace=False)
        balanced_idx = np.concatenate([pos_sampled, neg_sampled])
        rng.shuffle(balanced_idx)
        
        final_compounds = [compound_graphs[i] for i in balanced_idx]
        final_proteins = final_proteins[balanced_idx]
        final_labels = final_labels[balanced_idx]
        print(f"Final Model Dataset Balanced to {len(final_labels)} samples (Pos: {min_count}, Neg: {min_count}).")
    
    full_dataset = ORLigandDataset(
        final_compounds, final_proteins, final_labels,
        protein_tokenizer, config.protein_max_len
    )
    
    full_loader = DataLoader(
        full_dataset, 
        batch_size=config.batch_size, 
        shuffle=True,
        collate_fn=collate_fn
    )
    
    final_model = CPIPredictor(
        config, 
        protein_vocab_size=protein_tokenizer.vocab_size,
        node_in_dim=node_in_dim
    ).to(device)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(
        final_model.parameters(), 
        lr=config.learning_rate, 
        weight_decay=config.weight_decay
    )
    
    for epoch in range(config.num_epochs):
        train_loss = train_model(final_model, full_loader, criterion, optimizer, device)
        
        if (epoch + 1) % 5 == 0:
            print(f'Full Training Epoch {epoch + 1}/{config.num_epochs}, Loss: {train_loss:.4f}')
    
    torch.save(final_model.state_dict(), 'final_model.pth')
    print("Saved final model trained on entire dataset.")

if __name__ == '__main__':
    main()