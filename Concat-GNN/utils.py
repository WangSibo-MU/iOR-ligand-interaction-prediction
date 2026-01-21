import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
import math

# Hyper-parameters
class Config:
    def __init__(self):
        self.protein_max_len = 480
        self.protein_embed_dim = 64
        self.gcn_hidden_dim = 256
        self.gcn_output_dim = 256
        self.gcn_num_layers = 3
        self.gcn_dropout = 0.3
        self.hidden_dim = 256
        self.n_heads = 8
        self.n_layers = 3
        self.dropout = 0.3
        self.batch_size = 32
        self.learning_rate = 1e-4
        self.num_epochs = 100
        self.k_folds = 5
        self.weight_decay = 1e-5
        self.patience = 20

class CharTokenizer:
    def __init__(self, chars):
        self.chars = chars
        self.char_to_idx = {c: i+1 for i, c in enumerate(chars)}  # 0 reserved for padding
        self.idx_to_char = {i+1: c for i, c in enumerate(chars)}
        self.vocab_size = len(chars) + 1  # +1 for padding
    
    def encode(self, seq, max_len):
        encoded = [self.char_to_idx.get(c, 0) for c in seq[:max_len]]
        encoded += [0] * (max_len - len(encoded))
        return encoded

class CompoundGNN(nn.Module):
    def __init__(self, node_in_dim, gcn_hidden_dim, gcn_output_dim, num_layers=3, dropout=0.3):
        super(CompoundGNN, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.conv1 = GCNConv(node_in_dim, gcn_hidden_dim)
        
        self.convs = nn.ModuleList()
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(gcn_hidden_dim, gcn_hidden_dim))
        
        self.conv_out = GCNConv(gcn_hidden_dim, gcn_output_dim)
        
        self.bns = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.bns.append(nn.BatchNorm1d(gcn_hidden_dim))
        
        self.bn_out = nn.BatchNorm1d(gcn_output_dim)
        
        self.fc = nn.Sequential(
            nn.Linear(gcn_output_dim, gcn_output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gcn_output_dim, gcn_output_dim)
        )
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        x = F.relu(self.conv1(x, edge_index))
        x = self.bns[0](x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        for i in range(self.num_layers - 2):
            x = F.relu(self.convs[i](x, edge_index))
            x = self.bns[i+1](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = F.relu(self.conv_out(x, edge_index))
        x = self.bn_out(x)
        
        x = global_mean_pool(x, batch)
        
        x = self.fc(x)
        
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
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

class ProteinTransformer(nn.Module):
    def __init__(self, config, protein_vocab_size):
        super(ProteinTransformer, self).__init__()
        self.protein_embedding = nn.Embedding(
            num_embeddings=protein_vocab_size,
            embedding_dim=config.protein_embed_dim,
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
        
        self.protein_dim_adjust = nn.Linear(config.protein_embed_dim, config.hidden_dim)
        
        self.protein_pos_encoder = PositionalEncoding(config.protein_embed_dim, config.dropout)
        
    def forward(self, protein):
        protein_embedded = self.protein_embedding(protein)
        protein_embedded = self.protein_pos_encoder(protein_embedded.permute(1, 0, 2))
        protein_encoded = self.protein_encoder(protein_embedded)
        
        protein_encoded = protein_encoded.permute(1, 0, 2)
        protein_encoded = self.protein_dim_adjust(protein_encoded)
        
        protein_feat = protein_encoded.mean(dim=1)
        
        return protein_feat

class CPIPredictor(nn.Module):
    def __init__(self, config, protein_vocab_size, node_in_dim):
        super(CPIPredictor, self).__init__()
        
        self.compound_gnn = CompoundGNN(
            node_in_dim=node_in_dim,
            gcn_hidden_dim=config.gcn_hidden_dim,
            gcn_output_dim=config.gcn_output_dim,
            num_layers=config.gcn_num_layers,
            dropout=config.gcn_dropout
        )
        
        self.protein_transformer = ProteinTransformer(
            config, 
            protein_vocab_size=protein_vocab_size
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(config.gcn_output_dim + config.hidden_dim, config.hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim//2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, compound_data, protein):
        compound_feat = self.compound_gnn(compound_data)
        
        protein_feat = self.protein_transformer(protein)
        
        combined = torch.cat([compound_feat, protein_feat], dim=1)
        fused = self.fusion(combined)

        output = self.classifier(fused)
        return output.squeeze()