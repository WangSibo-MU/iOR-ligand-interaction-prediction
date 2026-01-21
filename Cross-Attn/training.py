import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, recall_score, matthews_corrcoef, roc_curve
import warnings
import os
from utils import Config, ORLigandDataset, CharTokenizer, ORLigandTransformer, PositionalEncoding, device

warnings.filterwarnings('ignore')

torch.manual_seed(42)
np.random.seed(42)

config = Config()

def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    
    all_labels = []
    for _, _, labels in train_loader:
        all_labels.extend(labels.numpy())
    all_labels = np.array(all_labels)
    
    
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
    train_data = np.load('processed_data/train.npz')
    smiles = train_data['smiles']
    proteins = train_data['proteins']
    labels = train_data['labels']
    
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

    kfold = StratifiedKFold(n_splits=config.k_folds, shuffle=True, random_state=42)
    results = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(encoded_smiles, labels)):
        print(f'\nFold {fold + 1}/{config.k_folds}')

        train_smiles = encoded_smiles[train_idx]
        train_proteins = encoded_proteins[train_idx]
        train_labels = labels[train_idx]
        
        val_smiles = encoded_smiles[val_idx]
        val_proteins = encoded_proteins[val_idx]
        val_labels = labels[val_idx]

        train_dataset = ORLigandDataset(train_smiles, train_proteins, train_labels)
        val_dataset = ORLigandDataset(val_smiles, val_proteins, val_labels)
        
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
    full_dataset = ORLigandDataset(encoded_smiles, encoded_proteins, labels)
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