import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math

class Config:
    def __init__(self):
        self.protein_max_len = 480
        self.smiles_embed_dim = 64
        self.protein_embed_dim = 64
        self.hidden_dim = 256
        self.n_heads = 8
        self.n_layers = 3
        self.dropout = 0.3
        self.batch_size = 32
        self.learning_rate = 1e-4
        self.num_epochs = 100
        self.k_folds = 5
        self.weight_decay = 1e-5

class CharTokenizer:
    def __init__(self, chars):
        self.chars = chars
        self.char_to_idx = {c: i+1 for i, c in enumerate(chars)}
        self.idx_to_char = {i+1: c for i, c in enumerate(chars)}
        self.vocab_size = len(chars) + 1
    
    def encode(self, seq, max_len):
        encoded = [self.char_to_idx.get(c, 0) for c in seq[:max_len]]
        encoded += [0] * (max_len - len(encoded))
        return encoded

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        pe = self.pe[:x.size(0), :]
        pe = pe.unsqueeze(1)
        x = x + pe
        return self.dropout(x)

class ProteinFeatureExtractor(nn.Module):
    def __init__(self, config, protein_vocab_size):
        super().__init__()
        self.protein_embedding = nn.Embedding(
            protein_vocab_size,
            config.protein_embed_dim,
            padding_idx=0
        )
        
        protein_encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.protein_embed_dim,
            nhead=config.n_heads,
            dim_feedforward=config.hidden_dim,
            dropout=config.dropout
        )
        self.protein_encoder = nn.TransformerEncoder(
            protein_encoder_layer,
            num_layers=config.n_layers
        )
        
        self.pos_encoder = PositionalEncoding(config.protein_embed_dim, config.dropout)
        self.dim_adjust = nn.Linear(config.protein_embed_dim, config.hidden_dim)
        self.norm = nn.LayerNorm(config.hidden_dim)
        
    def forward(self, protein):
        embedded = self.protein_embedding(protein)
        embedded = self.pos_encoder(embedded.permute(1, 0, 2))
        encoded = self.protein_encoder(embedded)
        encoded = self.dim_adjust(encoded)
        encoded = self.norm(encoded)
        encoded = encoded.mean(dim=0)
        return encoded

class LigandFeatureProcessor(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, output_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.3)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class CPIPredictor(nn.Module):
    def __init__(self, config, protein_vocab_size, ligand_input_dim):
        super().__init__()
        self.protein_extractor = ProteinFeatureExtractor(config, protein_vocab_size)
        self.ligand_processor = LigandFeatureProcessor(ligand_input_dim)
        
        self.classifier = nn.Sequential(
            nn.Linear(128 + config.hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
    def forward(self, protein, ligand_features):
        protein_features = self.protein_extractor(protein)
        ligand_features = self.ligand_processor(ligand_features)
        combined = torch.cat([protein_features, ligand_features], dim=1)
        return self.classifier(combined)

class CPIDataset(Dataset):
    def __init__(self, proteins, ligand_features, labels=None):
        self.proteins = proteins
        self.ligand_features = ligand_features
        self.labels = labels
        
    def __len__(self):
        return len(self.proteins)
    
    def __getitem__(self, idx):
        if self.labels is not None:
            return self.proteins[idx], self.ligand_features[idx], self.labels[idx]
        return self.proteins[idx], self.ligand_features[idx]