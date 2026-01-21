import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, recall_score, matthews_corrcoef, roc_curve
import warnings
warnings.filterwarnings('ignore')
import os
import pickle
from utils import Config, CharTokenizer, CompoundGNN, ProteinTransformer, CPIPredictor

warnings.filterwarnings('ignore')
os.environ["LOKY_MAX_CPU_COUNT"] = "4" 
torch.manual_seed(42)
np.random.seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = Config()

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
    print("Loading processed data...")
    try:
        compound_graphs = torch.load('processed_data/train_compounds.pt')
        
        train_data = np.load('processed_data/train_data.npz')
        proteins = train_data['proteins']
        labels = train_data['labels']
        
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
    
    kfold = StratifiedKFold(n_splits=config.k_folds, shuffle=True, random_state=42)
    results = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(compound_graphs, labels)):
        print(f'\nFold {fold + 1}/{config.k_folds}')
        
        train_compounds = [compound_graphs[i] for i in train_idx]
        train_proteins = proteins[train_idx]
        train_labels = labels[train_idx]
        
        val_compounds = [compound_graphs[i] for i in val_idx]
        val_proteins = proteins[val_idx]
        val_labels = labels[val_idx]
        
        train_dataset = ORLigandDataset(
            train_compounds, train_proteins, train_labels,
            protein_tokenizer, config.protein_max_len
        )
        
        val_dataset = ORLigandDataset(
            val_compounds, val_proteins, val_labels,
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
    
    full_dataset = ORLigandDataset(
        compound_graphs, proteins, labels,
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