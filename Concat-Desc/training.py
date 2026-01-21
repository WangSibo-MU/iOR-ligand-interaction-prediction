import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
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

    kfold = StratifiedKFold(n_splits=config.k_folds, shuffle=True, random_state=42)
    results_nn = []
    all_nn_preds = np.zeros(len(labels))

    for fold, (train_idx, val_idx) in enumerate(kfold.split(encoded_proteins, labels)):
        print(f'\nFold {fold + 1}/{config.k_folds}')

        train_proteins = encoded_proteins[train_idx]
        train_ligand = ligand_features[train_idx]
        train_labels = labels[train_idx]
        
        val_proteins = encoded_proteins[val_idx]
        val_ligand = ligand_features[val_idx]
        val_labels = labels[val_idx]

        train_dataset = CPIDataset(train_proteins, train_ligand, train_labels)
        val_dataset = CPIDataset(val_proteins, val_ligand, val_labels)
        
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

        all_nn_preds[val_idx] = preds

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

    full_dataset = CPIDataset(encoded_proteins, ligand_features, labels)
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