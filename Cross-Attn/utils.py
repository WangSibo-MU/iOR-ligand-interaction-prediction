import torch
import torch.nn as nn
import math
import numpy as np
from torch.utils.data import Dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Config:
    def __init__(self):
        self.smiles_max_len = 70
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

class ORLigandDataset(Dataset):
    def __init__(self, smiles, proteins, labels=None):
        self.smiles = smiles
        self.proteins = proteins
        self.labels = labels
        
    def __len__(self):
        return len(self.smiles)
    
    def __getitem__(self, idx):
        if self.labels is not None:
            return self.smiles[idx], self.proteins[idx], self.labels[idx]
        return self.smiles[idx], self.proteins[idx]

class CharTokenizer:
    def __init__(self, chars):
        self.chars = chars
        self.char_to_idx = {c: i+1 for i, c in enumerate(chars)}  # 0 reserved for padding
        self.idx_to_char = {i+1: c for i, c in enumerate(chars)}
        self.vocab_size = len(chars) + 1  # +1 for padding
    
    def encode(self, seq, max_len):
        if not isinstance(seq, str):
            seq = str(seq)
        
        encoded = [self.char_to_idx.get(c, 0) for c in seq[:max_len]]
        encoded += [0] * (max_len - len(encoded))
        return encoded

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

class CustomTransformerDecoderLayer(nn.TransformerDecoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(CustomTransformerDecoderLayer, self).__init__(d_model, nhead, dim_feedforward, dropout, activation)
        
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        tgt2, self_attn_weights = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                                                key_padding_mask=tgt_key_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        
        tgt2, cross_attn_weights = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                                      key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        
        return tgt, cross_attn_weights

class CustomTransformerDecoder(nn.TransformerDecoder):
    def __init__(self, decoder_layer, num_layers):
        super(CustomTransformerDecoder, self).__init__(decoder_layer, num_layers)
        
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        output = tgt
        attn_weights = None
        
        for mod in self.layers:
            output, attn_weights = mod(output, memory, tgt_mask=tgt_mask,
                                     memory_mask=memory_mask,
                                     tgt_key_padding_mask=tgt_key_padding_mask,
                                     memory_key_padding_mask=memory_key_padding_mask)
        
        return output, attn_weights

class ORLigandTransformer(nn.Module):
    def __init__(self, config, smiles_vocab_size, protein_vocab_size):
        super(ORLigandTransformer, self).__init__()

        self.smiles_embedding = nn.Embedding(
            num_embeddings=smiles_vocab_size,
            embedding_dim=config.smiles_embed_dim,
            padding_idx=0
        )

        self.protein_embedding = nn.Embedding(
            num_embeddings=protein_vocab_size,
            embedding_dim=config.protein_embed_dim,
            padding_idx=0
        )

        smiles_encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.smiles_embed_dim,
            nhead=config.n_heads,
            dim_feedforward=config.hidden_dim,
            dropout=config.dropout
        )
        self.smiles_encoder = nn.TransformerEncoder(
            smiles_encoder_layer,
            num_layers=config.n_layers
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

        self.smiles_dim_adjust = nn.Linear(config.smiles_embed_dim, config.hidden_dim)
        self.protein_dim_adjust = nn.Linear(config.protein_embed_dim, config.hidden_dim)

        self.smiles_norm = nn.LayerNorm(config.hidden_dim)
        self.protein_norm = nn.LayerNorm(config.hidden_dim)

        decoder_layer = CustomTransformerDecoderLayer(
            d_model=config.hidden_dim,
            nhead=config.n_heads,
            dim_feedforward=config.hidden_dim,
            dropout=config.dropout
        )
        self.decoder = CustomTransformerDecoder(
            decoder_layer,
            num_layers=config.n_layers * 2
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
        
        self.smiles_pos_encoder = PositionalEncoding(config.smiles_embed_dim, config.dropout)
        self.protein_pos_encoder = PositionalEncoding(config.protein_embed_dim, config.dropout)
        
    def forward(self, smiles, protein):
        smiles_embedded = self.smiles_embedding(smiles)
        smiles_embedded = self.smiles_pos_encoder(smiles_embedded.permute(1, 0, 2))
        smiles_encoded = self.smiles_encoder(smiles_embedded)

        protein_embedded = self.protein_embedding(protein)
        protein_embedded = self.protein_pos_encoder(protein_embedded.permute(1, 0, 2))
        protein_encoded = self.protein_encoder(protein_embedded)

        smiles_encoded = self.smiles_dim_adjust(smiles_encoded)
        protein_encoded = self.protein_dim_adjust(protein_encoded)

        smiles_encoded = self.smiles_norm(smiles_encoded)
        protein_encoded = self.protein_norm(protein_encoded)

        memory = protein_encoded
        tgt = smiles_encoded
        decoded, attention_weights = self.decoder(tgt, memory)

        decoded = decoded.mean(dim=0)

        output = self.classifier(decoded)
        return output.squeeze(), attention_weights