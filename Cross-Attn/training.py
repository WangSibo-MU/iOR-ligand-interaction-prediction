import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, recall_score, matthews_corrcoef
import warnings
import os
from utils import Config, ORLigandDataset, CharTokenizer, ORLigandTransformer, PositionalEncoding, device

warnings.filterwarnings('ignore')

# 固定随机种子以保证实验结果可复现
torch.manual_seed(42)
np.random.seed(42)

config = Config()

def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    
    for batch_idx, (smiles, protein, labels) in enumerate(train_loader):
        smiles = smiles.to(device)
        protein = protein.to(device)
        labels = labels.float().to(device)
        
        optimizer.zero_grad()
        outputs, _ = model(smiles, protein)
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

def evaluate_model(model, data_loader, device):
    model.eval()
    predictions = []
    true_labels = []
    with torch.no_grad():
        for smiles, protein, labels in data_loader:
            smiles = smiles.to(device)
            protein = protein.to(device)
            outputs, _ = model(smiles, protein)
            
            outputs_np = outputs.cpu().numpy()
            if outputs_np.ndim == 0:
                outputs_np = np.array([outputs_np])
                
            predictions.extend(outputs_np)
            true_labels.extend(labels.numpy())
    
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
    parser = argparse.ArgumentParser(description="Train OR-Ligand Transformer with CV")
    parser.add_argument('--ligand_cold_start', action='store_true', help='Enable ligand cold start for cross-validation (SMILES group split).')
    parser.add_argument('--protein_cold_start', action='store_true', help='Enable protein cold start for cross-validation (Protein sequence group split).')
    parser.add_argument('--balance_samples', action='store_true', help='Force equal number of positive and negative samples via downsampling.')
    args = parser.parse_args()

    # 互斥安全检查（配体冷启动与蛋白冷启动不可同时开启）
    if args.ligand_cold_start and args.protein_cold_start:
        raise ValueError("Cannot enable both ligand cold start and protein cold start at the same time. Please choose one.")

    # 读取经 data_processing.py 处理后保存的训练集数据
    train_data = np.load('processed_data/train.npz')
    smiles = train_data['smiles']
    proteins = train_data['proteins']
    labels = train_data['labels']

    # 建立 Tokenizer
    all_smiles_chars = set()
    for s in smiles:
        all_smiles_chars.update(s)
    smiles_tokenizer = CharTokenizer(sorted(all_smiles_chars))
    
    all_protein_chars = set()
    for p in proteins:
        all_protein_chars.update(p)
    protein_tokenizer = CharTokenizer(sorted(all_protein_chars))

    os.makedirs('models', exist_ok=True)
    with open('models/protein_tokenizer_chars.txt', 'w') as f:
        f.write(''.join(protein_tokenizer.chars))
    print("Saved protein tokenizer chars to models/protein_tokenizer_chars.txt")
    
    encoded_smiles = np.array([smiles_tokenizer.encode(s, config.smiles_max_len) for s in smiles])
    encoded_proteins = np.array([protein_tokenizer.encode(p, config.protein_max_len) for p in proteins])

    encoded_smiles = torch.LongTensor(encoded_smiles)
    encoded_proteins = torch.LongTensor(encoded_proteins)
    labels = torch.FloatTensor(labels)

    results = []
    
    # ================= 交叉验证策略分发 =================
    if args.ligand_cold_start:
        print("\n[Cold Start Option] Using StratifiedGroupKFold (Ligand Cold Start) for Cross-Validation.")
        # 使用去除 padding 的 SMILES 字符串作为 Group 键
        groups = np.array([s.strip() for s in smiles])
        kfold = StratifiedGroupKFold(n_splits=config.k_folds, shuffle=True, random_state=42)
        split_iterator = kfold.split(encoded_smiles, labels, groups)
    elif args.protein_cold_start:
        print("\n[Cold Start Option] Using StratifiedGroupKFold (Protein Cold Start) for Cross-Validation.")
        # 使用去除 padding 的蛋白氨基酸序列作为 Group 键
        groups = np.array([p.strip() for p in proteins])
        kfold = StratifiedGroupKFold(n_splits=config.k_folds, shuffle=True, random_state=42)
        split_iterator = kfold.split(encoded_smiles, labels, groups)
    else:
        print("\n[Default Mode] Using Random Stratified (StratifiedKFold) for Cross-Validation.")
        kfold = StratifiedKFold(n_splits=config.k_folds, shuffle=True, random_state=42)
        split_iterator = kfold.split(encoded_smiles, labels)

    for fold, (train_idx, val_idx) in enumerate(split_iterator):
        print(f'\nFold {fold + 1}/{config.k_folds}')
        
        # 提取当前折（Fold）的未处理原始子集
        fold_train_smiles = encoded_smiles[train_idx]
        fold_train_proteins = encoded_proteins[train_idx]
        fold_train_labels = labels[train_idx]

        fold_val_smiles = encoded_smiles[val_idx]
        fold_val_proteins = encoded_proteins[val_idx]
        fold_val_labels = labels[val_idx]

        # ================= 折内动态平衡机制（保证配体/蛋白完全隔离的同时正负样本绝对 1:1） =================
        if args.balance_samples:
            # 1. 训练集严格 1:1 下采样
            train_labels_np = fold_train_labels.numpy()
            pos_train_idx = np.where(train_labels_np == 1)[0]
            neg_train_idx = np.where(train_labels_np == 0)[0]
            train_min_count = min(len(pos_train_idx), len(neg_train_idx))
            
            rng = np.random.default_rng(42 + fold)  # 每个 Fold 使用不同的随机偏移种子
            pos_train_sampled = rng.choice(pos_train_idx, train_min_count, replace=False)
            neg_train_sampled = rng.choice(neg_train_idx, train_min_count, replace=False)
            
            balanced_train_idx = np.concatenate([pos_train_sampled, neg_train_sampled])
            rng.shuffle(balanced_train_idx)
            
            fold_train_smiles = fold_train_smiles[balanced_train_idx]
            fold_train_proteins = fold_train_proteins[balanced_train_idx]
            fold_train_labels = fold_train_labels[balanced_train_idx]

            # 2. 验证集严格 1:1 下采样
            val_labels_np = fold_val_labels.numpy()
            pos_val_idx = np.where(val_labels_np == 1)[0]
            neg_val_idx = np.where(val_labels_np == 0)[0]
            val_min_count = min(len(pos_val_idx), len(neg_val_idx))
            
            pos_val_sampled = rng.choice(pos_val_idx, val_min_count, replace=False)
            neg_val_sampled = rng.choice(neg_val_idx, val_min_count, replace=False)
            
            balanced_val_idx = np.concatenate([pos_val_sampled, neg_val_sampled])
            rng.shuffle(balanced_val_idx)
            
            fold_val_smiles = fold_val_smiles[balanced_val_idx]
            fold_val_proteins = fold_val_proteins[balanced_val_idx]
            fold_val_labels = fold_val_labels[balanced_val_idx]

        # 打印当前 Fold 实际输入给 DataLoader 的严格对等样本分布
        print(f"  - [Final Balanced] Train Set Size: {len(fold_train_labels)} (Pos: {int(torch.sum(fold_train_labels==1))}, Neg: {int(torch.sum(fold_train_labels==0))})")
        print(f"  - [Final Balanced] Val Set Size:   {len(fold_val_labels)} (Pos: {int(torch.sum(fold_val_labels==1))}, Neg: {int(torch.sum(fold_val_labels==0))})")

        train_dataset = ORLigandDataset(fold_train_smiles, fold_train_proteins, fold_train_labels)
        val_dataset = ORLigandDataset(fold_val_smiles, fold_val_proteins, fold_val_labels)
        
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

        model = ORLigandTransformer(
            config, 
            smiles_tokenizer.vocab_size, 
            protein_tokenizer.vocab_size
        ).to(device)
        
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), 
                              lr=config.learning_rate, 
                              weight_decay=config.weight_decay)

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

            if auc > best_auc:
                best_auc = auc
                best_acc = acc
                best_f1 = f1
                best_recall = recall
                best_mcc = mcc
                best_model_state = model.state_dict()
                
            if (epoch + 1) % 5 == 0:
                print(f'Epoch {epoch + 1}/{config.num_epochs}, Loss: {train_loss:.4f}, '
                      f'Val AUC: {auc:.4f}, Acc: {acc:.4f}, F1: {f1:.4f}, Recall: {recall:.4f}, MCC: {mcc:.4f}')

        print(f'Fold {fold + 1} Best Results - AUC: {best_auc:.4f}, '
              f'Acc: {best_acc:.4f}, F1: {best_f1:.4f}, '
              f'Recall: {best_recall:.4f}, MCC: {best_mcc:.4f}')       
        results.append({
            'auc': best_auc,
            'acc': best_acc,
            'f1': best_f1,
            'recall': best_recall,
            'mcc': best_mcc
        })

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
    
    # 最终模型训练也需要根据是否平衡进行样本下采样
    final_smiles, final_proteins, final_labels = encoded_smiles, encoded_proteins, labels
    if args.balance_samples:
        labels_np = labels.numpy()
        pos_idx = np.where(labels_np == 1)[0]
        neg_idx = np.where(labels_np == 0)[0]
        min_count = min(len(pos_idx), len(neg_idx))
        
        rng = np.random.default_rng(42)
        pos_sampled = rng.choice(pos_idx, min_count, replace=False)
        neg_sampled = rng.choice(neg_idx, min_count, replace=False)
        
        balanced_idx = np.concatenate([pos_sampled, neg_sampled])
        rng.shuffle(balanced_idx)
        
        final_smiles = final_smiles[balanced_idx]
        final_proteins = final_proteins[balanced_idx]
        final_labels = final_labels[balanced_idx]
        print(f"Final Model Dataset Balanced to {len(final_labels)} samples (Pos: {min_count}, Neg: {min_count}).")

    full_dataset = ORLigandDataset(final_smiles, final_proteins, final_labels)
    full_loader = DataLoader(full_dataset, batch_size=config.batch_size, shuffle=True)

    final_model = ORLigandTransformer(
        config, 
        smiles_tokenizer.vocab_size, 
        protein_tokenizer.vocab_size
    ).to(device)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(final_model.parameters(), 
                          lr=config.learning_rate, 
                          weight_decay=config.weight_decay)

    for epoch in range(config.num_epochs):
        train_loss = train_model(final_model, full_loader, criterion, optimizer, device)
        if (epoch + 1) % 5 == 0:
            print(f'Full Training Epoch {epoch + 1}/{config.num_epochs}, Loss: {train_loss:.4f}')

    torch.save(final_model.state_dict(), 'final_model.pth')
    print("Saved final model trained on entire dataset.")

if __name__ == '__main__':
    main()