import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
# 引入 StratifiedKFold 与 StratifiedGroupKFold 实现冷启动下的均衡交叉验证
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, recall_score, matthews_corrcoef, roc_curve
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
import os
from utils import Config, CharTokenizer, CPIDataset, CPIPredictor

warnings.filterwarnings('ignore')
os.environ["LOKY_MAX_CPU_COUNT"] = "4" 
torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config = Config()

os.makedirs('models', exist_ok=True)

def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    
    for batch_idx, (protein, ligand_features, labels) in enumerate(train_loader):
        protein = protein.to(device)
        ligand_features = ligand_features.float().to(device)
        labels = labels.float().to(device)
        
        optimizer.zero_grad()
        outputs = model(protein, ligand_features)
        outputs = outputs.squeeze(-1)
        
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
        for protein, ligand_features, labels in data_loader:
            protein = protein.to(device)
            ligand_features = ligand_features.float().to(device)
            outputs = model(protein, ligand_features)
            
            outputs = outputs.squeeze(-1)
            predictions.extend(outputs.cpu().numpy())
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
    
    return auc, acc, f1, recall, mcc, predictions, true_labels

def main():
    # ================= 命令行参数解析 =================
    parser = argparse.ArgumentParser(description="Train CPI Predictor with Cold Start and CV options.")
    parser.add_argument('--ligand_cold_start', action='store_true', help='Enable ligand cold start for cross-validation.')
    parser.add_argument('--protein_cold_start', action='store_true', help='Enable protein cold start for cross-validation.')
    parser.add_argument('--balance_samples', action='store_true', help='Force equal number of positive and negative samples via downsampling within folds.')
    args = parser.parse_args()

    if args.ligand_cold_start and args.protein_cold_start:
        raise ValueError("Cannot enable both ligand cold start and protein cold start at the same time. Please choose one.")

    print("="*60)
    print("Model training...")
    print("="*60)
    
    print("Loading preprocessed data...")
    train_data = np.load('processed_data/train.npz', allow_pickle=True)
    smiles = train_data['smiles']
    proteins = train_data['proteins']
    labels = train_data['labels']
    ligand_features = train_data['descriptors']
    
    print(f"Loading data: {len(smiles)} samples")
    print(f"Descriptor dimension: {ligand_features.shape[1]}")

    with open('models/protein_tokenizer_chars.txt', 'r') as f:
        protein_chars = f.read().strip().split(',')
    protein_tokenizer = CharTokenizer(protein_chars)
    print(f"Loading protein tokenizer, including {len(protein_chars)} tokens")

    print("Standardizing molecular descriptors...")
    scaler = StandardScaler()
    ligand_features = scaler.fit_transform(ligand_features)
    joblib.dump(scaler, 'models/ligand_scaler.pkl')
    print(f"Standardization completed, save the standardizer to models/ligand_scaler.pkl")

    encoded_proteins = np.array([protein_tokenizer.encode(p, config.protein_max_len) for p in proteins])
    encoded_proteins = torch.LongTensor(encoded_proteins)
    ligand_features = torch.FloatTensor(ligand_features)
    labels = torch.FloatTensor(labels)

    # ================= 交叉验证策略分发 =================
    if args.ligand_cold_start:
        print("\n[Cold Start Option] Using StratifiedGroupKFold (Ligand Cold Start).")
        groups = np.array([str(s).strip() for s in smiles])
        kfold = StratifiedGroupKFold(n_splits=config.k_folds, shuffle=True, random_state=42)
        split_iterator = kfold.split(encoded_proteins, labels, groups)
    elif args.protein_cold_start:
        print("\n[Cold Start Option] Using StratifiedGroupKFold (Protein Cold Start).")
        groups = np.array([str(p).strip() for p in proteins])
        kfold = StratifiedGroupKFold(n_splits=config.k_folds, shuffle=True, random_state=42)
        split_iterator = kfold.split(encoded_proteins, labels, groups)
    else:
        print("\n[Default Mode] Using Random Stratified (StratifiedKFold).")
        kfold = StratifiedKFold(n_splits=config.k_folds, shuffle=True, random_state=42)
        split_iterator = kfold.split(encoded_proteins, labels)

    results_nn = []
    all_nn_preds = np.zeros(len(labels))

    for fold, (train_idx, val_idx) in enumerate(split_iterator):
        print(f'\nFold {fold + 1}/{config.k_folds}')

        # 提取当前 Fold 的原始数据
        fold_train_proteins = encoded_proteins[train_idx]
        fold_train_ligand = ligand_features[train_idx]
        fold_train_labels = labels[train_idx]
        
        fold_val_proteins = encoded_proteins[val_idx]
        fold_val_ligand = ligand_features[val_idx]
        fold_val_labels = labels[val_idx]

        # ================= 折内动态平衡机制 =================
        if args.balance_samples:
            # 1. 训练集严格 1:1
            train_labels_np = fold_train_labels.numpy()
            pos_train_idx = np.where(train_labels_np == 1)[0]
            neg_train_idx = np.where(train_labels_np == 0)[0]
            train_min_count = min(len(pos_train_idx), len(neg_train_idx))
            
            rng = np.random.default_rng(42 + fold)
            pos_train_sampled = rng.choice(pos_train_idx, train_min_count, replace=False)
            neg_train_sampled = rng.choice(neg_train_idx, train_min_count, replace=False)
            balanced_train_idx = np.concatenate([pos_train_sampled, neg_train_sampled])
            rng.shuffle(balanced_train_idx)
            
            fold_train_proteins = fold_train_proteins[balanced_train_idx]
            fold_train_ligand = fold_train_ligand[balanced_train_idx]
            fold_train_labels = fold_train_labels[balanced_train_idx]

            # 2. 验证集严格 1:1
            val_labels_np = fold_val_labels.numpy()
            pos_val_idx = np.where(val_labels_np == 1)[0]
            neg_val_idx = np.where(val_labels_np == 0)[0]
            val_min_count = min(len(pos_val_idx), len(neg_val_idx))
            
            pos_val_sampled = rng.choice(pos_val_idx, val_min_count, replace=False)
            neg_val_sampled = rng.choice(neg_val_idx, val_min_count, replace=False)
            balanced_val_idx = np.concatenate([pos_val_sampled, neg_val_sampled])
            rng.shuffle(balanced_val_idx)
            
            fold_val_proteins = fold_val_proteins[balanced_val_idx]
            fold_val_ligand = fold_val_ligand[balanced_val_idx]
            fold_val_labels = fold_val_labels[balanced_val_idx]

        # 打印各 Fold 真实参与训练和验证的样本分布
        print(f"  - [Final Balanced] Train Size: {len(fold_train_labels)} (Pos: {int(np.sum(fold_train_labels.numpy()==1))}, Neg: {int(np.sum(fold_train_labels.numpy()==0))})")
        print(f"  - [Final Balanced] Val Size:   {len(fold_val_labels)} (Pos: {int(np.sum(fold_val_labels.numpy()==1))}, Neg: {int(np.sum(fold_val_labels.numpy()==0))})")

        train_dataset = CPIDataset(fold_train_proteins, fold_train_ligand, fold_train_labels)
        val_dataset = CPIDataset(fold_val_proteins, fold_val_ligand, fold_val_labels)
        
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

        model = CPIPredictor(
            config, 
            protein_tokenizer.vocab_size, 
            ligand_features.shape[1]
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
        
        for epoch in range(config.num_epochs):
            train_loss = train_model(model, train_loader, criterion, optimizer, device)
            auc, acc, f1, recall, mcc, preds, true_labels = evaluate_model(model, val_loader, device)

            scheduler.step(auc)
            
            if auc > best_auc:
                best_auc = auc
                best_acc = acc
                best_f1 = f1
                best_recall = recall
                best_mcc = mcc
                best_model_state = model.state_dict()
                best_preds = preds
            
            if (epoch + 1) % 5 == 0:
                print(f'Epoch {epoch + 1}/{config.num_epochs}, Loss: {train_loss:.4f}, '
                      f'Val AUC: {auc:.4f}, Acc: {acc:.4f}, F1: {f1:.4f}, Recall: {recall:.4f}, MCC: {mcc:.4f}')
        
        print(f'Fold {fold + 1} Best Results - AUC: {best_auc:.4f}, Acc: {best_acc:.4f}, '
              f'F1: {best_f1:.4f}, Recall: {best_recall:.4f}, MCC: {best_mcc:.4f}')
        
        results_nn.append({
            'auc': best_auc,
            'acc': best_acc,
            'f1': best_f1,
            'recall': best_recall,
            'mcc': best_mcc
        })

        if args.balance_samples:
            pass  # 如果经过动态平衡降采样，索引已脱离原 val_idx 的对应关系，忽略全局写入
        else:
            all_nn_preds[val_idx] = best_preds

    avg_auc_nn = np.mean([r['auc'] for r in results_nn])
    avg_acc_nn = np.mean([r['acc'] for r in results_nn])
    avg_f1_nn = np.mean([r['f1'] for r in results_nn])
    avg_recall_nn = np.mean([r['recall'] for r in results_nn])
    avg_mcc_nn = np.mean([r['mcc'] for r in results_nn])
    
    print('\n' + "="*60)
    print("5-Fold Cross Validation Average Results:")
    print(f"AUC: {avg_auc_nn:.4f}")
    print(f"Accuracy: {avg_acc_nn:.4f}")
    print(f"F1 Score: {avg_f1_nn:.4f}")
    print(f"Recall: {avg_recall_nn:.4f}")
    print(f"MCC: {avg_mcc_nn:.4f}")
    print("="*60)

    print("\nTraining final model...")

    # 最终模型训练的数据平衡
    final_proteins = encoded_proteins
    final_ligands = ligand_features
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
        
        final_proteins = final_proteins[balanced_idx]
        final_ligands = final_ligands[balanced_idx]
        final_labels = final_labels[balanced_idx]
        print(f"Final Model Dataset Balanced to {len(final_labels)} samples (Pos: {min_count}, Neg: {min_count}).")

    full_dataset = CPIDataset(final_proteins, final_ligands, final_labels)
    full_loader = DataLoader(full_dataset, batch_size=config.batch_size, shuffle=True)
    
    final_nn_model = CPIPredictor(
        config, 
        protein_tokenizer.vocab_size, 
        ligand_features.shape[1]
    ).to(device)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(final_nn_model.parameters(), 
                          lr=config.learning_rate, 
                          weight_decay=config.weight_decay)

    for epoch in range(config.num_epochs):
        train_loss = train_model(final_nn_model, full_loader, criterion, optimizer, device)

        if (epoch + 1) % 5 == 0:
            print(f'Full Training Epoch {epoch + 1}/{config.num_epochs}, Loss: {train_loss:.4f}')

    torch.save(final_nn_model.state_dict(), 'models/final_nn_model.pth')
    print("\nSaving final model: models/final_nn_model.pth")
    print("="*60)

if __name__ == '__main__':
    main()