import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, recall_score, matthews_corrcoef, roc_curve, confusion_matrix
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns
import warnings
import os
from utils import Config, ORLigandDataset, CharTokenizer, ORLigandTransformer, PositionalEncoding, device
import pandas as pd
import matplotlib.font_manager as fm
import string
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import SimilarityMaps
import matplotlib.colors as mcolors
from PIL import Image, ImageDraw, ImageFont
import io
import re
import random


def set_global_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

set_global_seed(42)
# ========================================================

os.environ["LOKY_MAX_CPU_COUNT"] = "4"
warnings.filterwarnings('ignore')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
config = Config()

# ===================== User Configuration Area =====================
TASK = 'validate'  # 'validate' or 'predict'
PREDICTION_FILE = 'text.txt'
OUTPUT_FILE = 'predictions.txt'
LOCAL_SAMPLE_INDICES = [608, 609, 610]
# =====================================================

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def set_chemical_font():
    font_options = ['DejaVu Sans', 'Arial', 'Liberation Sans', 'sans-serif']
    for font in font_options:
        if fm.findfont(fm.FontProperties(family=font)):
            plt.rcParams['font.family'] = font
            break
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.unicode_minus'] = False

set_chemical_font()

class CharExplainer:
    @staticmethod
    def get_char_description(char_code):
        if 32 <= char_code <= 126:
            char = chr(char_code)
            if char in string.ascii_letters:
                return f"'{char}' (Letter)"
            elif char in string.digits:
                return f"'{char}' (Digit)"
            elif char in string.punctuation:
                return f"'{char}' (Punctuation)"
            elif char.isspace():
                return f"'{char}' (Whitespace)"
            else:
                return f"'{char}' (Symbol)"
        
        chem_symbols = {
            0: 'PAD',
            1: 'UNK',
        }
        
        if char_code in chem_symbols:
            return chem_symbols[char_code]
        
        return f"Char 0x{char_code:02X}"

class SMILESAttentionExplainer:
    def __init__(self, model, device, smiles_tokenizer, protein_tokenizer):
        self.model = model
        self.device = device
        self.smiles_tokenizer = smiles_tokenizer
        self.protein_tokenizer = protein_tokenizer
        self.model.eval()
    
    def extract_attention(self, smiles_tensor, protein_tensor):
        smiles_tensor = smiles_tensor.clone().detach().to(self.device)
        protein_tensor = protein_tensor.clone().detach().to(self.device)
        
        with torch.no_grad():
            output, attention_weights = self.model(smiles_tensor, protein_tensor)
        
        if attention_weights is not None:
            if attention_weights.dim() == 4:
                avg_attention = attention_weights.mean(dim=1)
                smiles_attention = avg_attention.mean(dim=2)
                cross_attention_matrix = avg_attention
            else:
                smiles_attention = attention_weights.mean(dim=1)
                cross_attention_matrix = attention_weights
            
            smiles_attention = smiles_attention.squeeze(0)
            cross_attention_matrix = cross_attention_matrix.squeeze(0)
            return smiles_attention.cpu().numpy(), cross_attention_matrix.cpu().numpy(), output.item()
        else:
            return None, None, output.item()
    
    def visualize_smiles_attention(self, smiles_str, attention_weights, output_path, 
                                 title="SMILES Attention", true_label=None, prediction=None):
        valid_chars = []
        valid_attention = []
        
        for i, char in enumerate(smiles_str):
            if char == ' ' or char.isspace() or char == '\x00':
                continue
                
            valid_chars.append(char)
            if i < len(attention_weights):
                valid_attention.append(attention_weights[i])
            else:
                valid_attention.append(0.0)
        
        if not valid_chars:
            print(f"Warning: No valid characters found for visualization. Skipping {output_path}")
            return
        
        full_title = title
        if true_label is not None and prediction is not None:
            full_title = f"{title}\nActual: {int(true_label)}, Prediction: {prediction:.4f}"
        elif prediction is not None:
            full_title = f"{title}\nPrediction: {prediction:.4f}"
        
        fig, ax = plt.subplots(figsize=(max(12, len(valid_chars) * 0.8), 6))
        
        norm = mcolors.Normalize(vmin=min(valid_attention), vmax=max(valid_attention))
        cmap = plt.cm.Reds
        
        for i, (char, attention) in enumerate(zip(valid_chars, valid_attention)):
            color = cmap(norm(attention))
            ax.text(i, 0.5, char, fontsize=15, ha='center', va='center', 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7))
                   
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Attention Weight', fontsize=15)
        
        ax.set_xlim(-0.5, len(valid_chars) - 0.5)
        ax.set_ylim(0, 1)
        ax.set_title(full_title, fontsize=17)
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"SMILES attention visualization saved to {output_path}")
    
    def map_char_attention_to_atoms(self, smiles_str, attention_weights):
        try:
            mol = Chem.MolFromSmiles(smiles_str)
            if mol is None:
                return None

            num_atoms = mol.GetNumAtoms()

            atom_attention = [0.0] * num_atoms
            atom_count = [0] * num_atoms

            atom_pattern = re.compile(r'([A-Z][a-z]?|\[[^\]]+\]|[a-z]|[\+\-]\d*|\d+)')
            matches = list(atom_pattern.finditer(smiles_str))

            for i, match in enumerate(matches):
                start, end = match.span()
                if i < num_atoms:
                    char_attention = []
                    for j in range(start, end):
                        if j < len(attention_weights):
                            char_attention.append(attention_weights[j])
                    
                    if char_attention:
                        atom_attention[i] = sum(char_attention) / len(char_attention)
                        atom_count[i] = len(char_attention)
            
            return atom_attention
        except Exception as e:
            print(f"Error in mapping char attention to atoms: {str(e)}")
            return None
    
    def visualize_molecule_attention(self, smiles_str, attention_weights, output_path, 
                                   title="Molecular Attention", true_label=None, prediction=None):
        try:
            mol = Chem.MolFromSmiles(smiles_str)
            if mol is None:
                print(f"Warning: Could not parse SMILES: {smiles_str}")
                return

            atom_weights = self.map_char_attention_to_atoms(smiles_str, attention_weights)
            
            if atom_weights is None:
                num_atoms = mol.GetNumAtoms()
                atom_weights = [sum(attention_weights) / len(attention_weights)] * num_atoms
            
            max_weight = max(atom_weights) if max(atom_weights) > 0 else 1.0
            min_weight = min(atom_weights) if min(atom_weights) < max_weight else 0.0
            norm_weights = [(w - min_weight) / (max_weight - min_weight) for w in atom_weights]
            
            full_title = title
            if true_label is not None and prediction is not None:
                full_title = f"{title}\nActual: {int(true_label)}, Prediction: {prediction:.4f}"
            elif prediction is not None:
                full_title = f"{title}\nPrediction: {prediction:.4f}"
            
            fig = plt.figure(figsize=(10, 8))
            
            try:
                from rdkit.Chem.Draw import MolDraw2DCairo
                drawer = MolDraw2DCairo(800, 800)
                
                SimilarityMaps.GetSimilarityMapFromWeights(
                    mol, norm_weights, 
                    draw2d=drawer,
                    colorMap=plt.cm.Reds,
                    contourLines=10,
                    coordScale=1.5,
                    alpha=0.0
                )

                drawer.WriteDrawingText(output_path)

                img = Image.open(output_path)

                fig, (ax_img, ax_cbar) = plt.subplots(1, 2, figsize=(12, 8), 
                                                     gridspec_kw={'width_ratios': [5, 1]})
                
                ax_img.imshow(np.array(img))
                ax_img.set_title(full_title, fontsize=19, pad=20)
                ax_img.axis('off')

                norm = mcolors.Normalize(vmin=min_weight, vmax=max_weight)
                sm = plt.cm.ScalarMappable(cmap=plt.cm.Reds, norm=norm)
                sm.set_array([])
                cbar = plt.colorbar(sm, cax=ax_cbar)
                cbar.set_label('Attention Weight', fontsize=15)

                plt.tight_layout()
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"High-resolution molecular attention visualization saved to {output_path}")
                
            except Exception as e:
                print(f"Error in molecular visualization with MolDraw2DCairo: {str(e)}")
                try:
                    img = SimilarityMaps.GetSimilarityMapFromWeights(
                        mol, norm_weights,
                        colorMap=plt.cm.Reds,
                        contourLines=10,
                        coordScale=1.5
                    )
                    
                    fig, (ax_img, ax_cbar) = plt.subplots(1, 2, figsize=(12, 8), 
                                                         gridspec_kw={'width_ratios': [5, 1]})
                    
                    ax_img.imshow(np.array(img))
                    ax_img.set_title(full_title, fontsize=19, pad=20)
                    ax_img.axis('off')
                    
                    norm = mcolors.Normalize(vmin=min_weight, vmax=max_weight)
                    sm = plt.cm.ScalarMappable(cmap=plt.cm.Reds, norm=norm)
                    sm.set_array([])
                    cbar = plt.colorbar(sm, cax=ax_cbar)
                    cbar.set_label('Attention Weight', fontsize=15)

                    plt.tight_layout()
                    plt.savefig(output_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    print(f"Alternative molecular attention visualization saved to {output_path}")
                    
                except Exception as e2:
                    print(f"Alternative method also failed: {str(e2)}")
                    try:
                        img = Draw.MolToImage(mol, size=(800, 800))

                        fig, ax = plt.subplots(figsize=(10, 8))
                        ax.imshow(np.array(img))
                        ax.set_title(full_title, fontsize=19, pad=20)
                        ax.axis('off')

                        plt.savefig(output_path, dpi=300, bbox_inches='tight')
                        plt.close()
                        
                        print(f"Fallback molecular image saved to {output_path}")
                    except Exception as e3:
                        print(f"Fallback also failed: {str(e3)}")
            
        except Exception as e:
            print(f"Error in molecular visualization: {str(e)}")
            try:
                mol = Chem.MolFromSmiles(smiles_str)
                if mol is not None:
                    img = Draw.MolToImage(mol, size=(800, 800))
                    plt.figure(figsize=(10, 8))
                    plt.imshow(np.array(img))
                    plt.title(full_title, fontsize=19, pad=20)
                    plt.axis('off')
                    plt.savefig(output_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    print(f"Fallback molecular image saved to {output_path}")
            except Exception as e2:
                print(f"Fallback also failed: {str(e2)}")
    
    def visualize_cross_attention_heatmap(self, smiles_str, protein_str, cross_attention_matrix, 
                                        output_path, title="Cross Attention Heatmap", 
                                        true_label=None, prediction=None, aa_per_row=60):
        try:
            valid_smiles_chars = []
            valid_protein_chars = []
            for i, char in enumerate(smiles_str):
                if char == ' ' or char.isspace() or char == '\x00' or i >= cross_attention_matrix.shape[0]:
                    continue
                valid_smiles_chars.append(char)
            for j, char in enumerate(protein_str):
                if char == ' ' or char.isspace() or char == '\x00' or j >= cross_attention_matrix.shape[1]:
                    continue
                valid_protein_chars.append(char)
            valid_matrix = cross_attention_matrix[:len(valid_smiles_chars), :len(valid_protein_chars)]
            
            if valid_matrix.size == 0:
                print(f"Warning: No valid attention matrix for visualization. Skipping {output_path}")
                return
            
            full_title = title
            if true_label is not None and prediction is not None:
                full_title = f"{title}\nActual: {int(true_label)}, Prediction: {prediction:.4f}"
            elif prediction is not None:
                full_title = f"{title}\nPrediction: {prediction:.4f}"

            n_rows = (len(valid_protein_chars) + aa_per_row - 1) // aa_per_row

            fig, axes = plt.subplots(n_rows, 1, figsize=(max(12, aa_per_row * 0.5), max(4 * n_rows, 6)))

            if n_rows == 1:
                axes = [axes]

            vmin, vmax = np.min(valid_matrix), np.max(valid_matrix)

            for row_idx in range(n_rows):
                start_idx = row_idx * aa_per_row
                end_idx = min((row_idx + 1) * aa_per_row, len(valid_protein_chars))

                current_protein_chars = valid_protein_chars[start_idx:end_idx]
                current_matrix = valid_matrix[:, start_idx:end_idx]

                im = axes[row_idx].imshow(current_matrix, cmap='Reds', aspect='auto', 
                                        interpolation='nearest', vmin=vmin, vmax=vmax)

                axes[row_idx].set_xticks(range(len(current_protein_chars)))
                axes[row_idx].set_xticklabels(current_protein_chars, fontsize=10, rotation=0)
                axes[row_idx].set_yticks(range(len(valid_smiles_chars)))
                axes[row_idx].set_yticklabels(valid_smiles_chars, fontsize=10)
                
                axes[row_idx].set_ylabel(f'AA {start_idx+1}-{end_idx}', fontsize=10)

                axes[row_idx].set_xticks(np.arange(-0.5, len(current_protein_chars), 1), minor=True)
                axes[row_idx].set_yticks(np.arange(-0.5, len(valid_smiles_chars), 1), minor=True)
                axes[row_idx].grid(which="minor", color="gray", linestyle='-', linewidth=0.1, alpha=0.3)

            cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
            cbar = fig.colorbar(im, cax=cbar_ax)
            cbar.set_label('Attention Weight', fontsize=15)

            fig.suptitle(full_title, fontsize=17, y=0.98)
            fig.text(0.5, 0.02, 'Protein Sequence', ha='center', fontsize=15)
            fig.text(0.02, 0.5, 'SMILES Sequence', va='center', rotation='vertical', fontsize=15)

            plt.tight_layout(rect=[0.03, 0.03, 0.9, 0.95])
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Cross attention heatmap saved to {output_path}")

            matrix_path = output_path.replace('.png', '_matrix.txt')
            np.savetxt(matrix_path, valid_matrix, fmt='%.6f')
            print(f"Cross attention matrix saved to {matrix_path}")
            
        except Exception as e:
            print(f"Error in cross attention visualization: {str(e)}")
    
    def explain(self, smiles_tensor, protein_tensor, smiles_str, protein_str):
        smiles_attention, cross_attention_matrix, prediction = self.extract_attention(smiles_tensor, protein_tensor)
        return smiles_attention, cross_attention_matrix, prediction, smiles_str, protein_str

def evaluate_model(model, data_loader, device, model_name="Final Model", explainer=None, local_sample_indices=None, smiles_strings=None, protein_strings=None):
    model.eval()
    predictions = []
    true_labels = []
    probabilities = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            if len(batch) == 3:
                smiles, protein, labels = batch
                smiles = smiles.to(device)
                protein = protein.to(device)
                true_labels.extend(labels.numpy())
            else:
                smiles, protein = batch
                smiles = smiles.to(device)
                protein = protein.to(device)
                
            outputs, _ = model(smiles, protein)
            
            probs = outputs.cpu().numpy().astype(float)
            if probs.ndim == 0:
                probs = np.array([probs])
                
            probabilities.extend(probs)
            predictions.extend((probs > 0.5).astype(int))

    probabilities = np.array(probabilities)
    predictions = np.array(predictions)

    if explainer is not None and local_sample_indices and smiles_strings is not None and protein_strings is not None and len(smiles_strings) > 0:
        os.makedirs(RESULTS_DIR, exist_ok=True)

        print(f"Generating specific explanations for {len(local_sample_indices)} samples...")

        all_smiles = []
        all_proteins = []
        all_labels = []
        
        for batch in data_loader:
            if len(batch) == 3:
                smiles, protein, labels = batch
                all_labels.extend(labels.numpy())
            else:
                smiles, protein = batch
                all_labels.extend([None] * len(smiles))
            
            all_smiles.append(smiles)
            all_proteins.append(protein)

        all_smiles = torch.cat(all_smiles, dim=0)
        all_proteins = torch.cat(all_proteins, dim=0)
        all_labels = np.array(all_labels)
        
        for idx in local_sample_indices:
            if idx < len(all_smiles):
                if idx < len(smiles_strings):
                    smiles_str = smiles_strings[idx]
                    protein_str = protein_strings[idx]
                else:
                    print(f"Warning: No SMILES/protein string available for sample {idx}")
                    continue

                smiles_attention, cross_attention_matrix, prediction = explainer.extract_attention(
                    all_smiles[idx].unsqueeze(0), 
                    all_proteins[idx].unsqueeze(0)
                )
                
                if smiles_attention is not None:
                    true_label = all_labels[idx] if all_labels[idx] is not None else None

                    explainer.visualize_smiles_attention(
                        smiles_str,
                        smiles_attention,
                        os.path.join(RESULTS_DIR, f'sample_{idx}_smiles_attention.png'),
                        title=f"SMILES Attention (Sample {idx})",
                        true_label=true_label,
                        prediction=prediction
                    )

                    explainer.visualize_molecule_attention(
                        smiles_str,
                        smiles_attention,
                        os.path.join(RESULTS_DIR, f'sample_{idx}_molecular_attention.png'),
                        title=f"Molecular Attention (Sample {idx})",
                        true_label=true_label,
                        prediction=prediction
                    )
                
                if cross_attention_matrix is not None:
                    explainer.visualize_cross_attention_heatmap(
                        smiles_str,
                        protein_str,
                        cross_attention_matrix,
                        os.path.join(RESULTS_DIR, f'sample_{idx}_cross_attention.png'),
                        title=f"Cross Attention (Sample {idx})",
                        true_label=true_label,
                        prediction=prediction
                    )
    
    if true_labels:
        true_labels = np.array(true_labels)
        auc = roc_auc_score(true_labels, probabilities) if len(np.unique(true_labels)) > 1 else 0.5
        acc = accuracy_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions) if len(np.unique(true_labels)) > 1 else 0
        recall = recall_score(true_labels, predictions) if np.sum(true_labels) > 0 else 0
        mcc = matthews_corrcoef(true_labels, predictions) if len(np.unique(true_labels)) > 1 else 0

        optimal_threshold = 0.5

        cm = confusion_matrix(true_labels, predictions)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Negative', 'Positive'], 
                    yticklabels=['Negative', 'Positive'])
        plt.ylabel('True Label', fontsize=15)
        plt.xlabel('Predicted Label', fontsize=15)
        plt.title('Confusion Matrix', fontsize=17)
        plt.savefig(os.path.join(RESULTS_DIR, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()

        with open(os.path.join(RESULTS_DIR, 'evaluation_results.txt'), 'w') as f:
            f.write(f"Model Name: {model_name}\n")
            f.write(f"Test Set Size: {len(true_labels)} samples\n")
            f.write(f"AUC: {auc:.4f}\n")
            f.write(f"Accuracy: {acc:.4f}\n")
            f.write(f"F1 Score: {f1:.4f}\n")
            f.write(f"Recall: {recall:.4f}\n")
            f.write(f"Matthews Correlation Coefficient (MCC): {mcc:.4f}\n")
            f.write(f"Threshold: {optimal_threshold:.4f} (Fixed at 0.5)\n")

        return {
            'auc': auc,
            'accuracy': acc,
            'f1': f1,
            'recall': recall,
            'mcc': mcc,
            'probabilities': probabilities,
            'true_labels': true_labels,
            'optimal_threshold': optimal_threshold
        }
    else:
        return probabilities

def create_global_explanation_plots(predictions, labels, results_dir=RESULTS_DIR):
    plt.figure(figsize=(10, 8))
    plt.hist(predictions, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel('Prediction Value')
    plt.ylabel('Frequency')
    plt.title('Distribution of Prediction Values')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(results_dir, 'prediction_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    if len(np.unique(labels)) > 1:
        plt.figure(figsize=(10, 8))
        unique_labels = np.unique(labels)
        for label in unique_labels:
            plt.hist(predictions[labels == label], bins=20, alpha=0.7, 
                     label=f'Class {int(label)}', density=True)
        plt.xlabel('Prediction Value')
        plt.ylabel('Density')
        plt.title('Prediction Distribution by Class')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(results_dir, 'prediction_by_class.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
    if len(np.unique(labels)) > 1:
        fpr, tpr, _ = roc_curve(labels, predictions)
        auc_score = roc_auc_score(labels, predictions)
        
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {auc_score:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(results_dir, 'roc_curve_global.png'), dpi=300, bbox_inches='tight')
        plt.close()

    if len(np.unique(labels)) > 1:
        optimal_threshold = 0.5
        binary_preds = (predictions > optimal_threshold).astype(int)

        acc = accuracy_score(labels, binary_preds)
        f1 = f1_score(labels, binary_preds)
        recall = recall_score(labels, binary_preds)
        mcc = matthews_corrcoef(labels, binary_preds)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axis('tight')
        ax.axis('off')
        
        metrics_data = [
            ['AUC', f'{auc_score:.4f}'],
            ['Accuracy', f'{acc:.4f}'],
            ['F1 Score', f'{f1:.4f}'],
            ['Recall', f'{recall:.4f}'],
            ['MCC', f'{mcc:.4f}'],
            ['Threshold', f'{optimal_threshold:.4f}']
        ]
        
        table = ax.table(cellText=metrics_data, 
                        colLabels=['Metric', 'Value'],
                        cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 2)
        ax.set_title('Performance Metrics', fontsize=17)
        
        plt.savefig(os.path.join(results_dir, 'performance_metrics.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    print("Global explanation plots saved to results directory")

def visualize_sequence_features(sequences, labels, results_dir=RESULTS_DIR):
    try:
        features = []
        for seq in sequences:
            char_counts = np.zeros(256)
            for char in seq:
                if ord(char) < 256:
                    char_counts[ord(char)] += 1
            features.append(char_counts / len(seq))
        
        features = np.array(features)

        tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=300, n_jobs=1)
        features_2d = tsne.fit_transform(features)

        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='viridis', alpha=0.7)
        plt.colorbar(scatter, label='Class Label')
        plt.title('t-SNE Visualization of Compound Features')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.grid(True, alpha=0.3)

        plt.savefig(os.path.join(results_dir, 'Compound_tsne.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("Sequence feature visualization saved")
    except Exception as e:
        print(f"Error in t-SNE visualization: {str(e)}")
        try:
            pca = PCA(n_components=2, random_state=42)
            features_2d = pca.fit_transform(features)
            
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='viridis', alpha=0.7)
            plt.colorbar(scatter, label='Class Label')
            plt.title('PCA Visualization of Sequence Features (t-SNE failed)')
            plt.xlabel('Principal Component 1')
            plt.ylabel('Principal Component 2')
            plt.grid(True, alpha=0.3)
            
            plt.savefig(os.path.join(results_dir, 'sequence_pca.png'), dpi=300, bbox_inches='tight')
            plt.close()
            print("PCA visualization saved as fallback")
        except Exception as e2:
            print(f"PCA visualization also failed: {str(e2)}")

def analyze_sequence_features(sequences, labels, results_dir=RESULTS_DIR):
    try:
        features = []
        char_frequencies = {}
        
        for seq in sequences:
            char_counts = np.zeros(256)
            for char in seq:
                char_code = ord(char)
                if char_code < 256:
                    char_counts[char_code] += 1
                    if char_code in char_frequencies:
                        char_frequencies[char_code] += 1
                    else:
                        char_frequencies[char_code] = 1
            features.append(char_counts / len(seq))
        
        features = np.array(features)

        unique_labels = np.unique(labels)
        mean_features = []
        
        for label in unique_labels:
            mean_feature = np.mean(features[labels == label], axis=0)
            mean_features.append(mean_feature)
        
        mean_features = np.array(mean_features)

        feature_variance = np.var(features, axis=0)
        top_features = np.argsort(feature_variance)[-10:]
        
        plt.figure(figsize=(12, 8))
        char_descriptions = [CharExplainer.get_char_description(i) for i in top_features]
        bars = plt.barh(range(10), feature_variance[top_features])
        plt.yticks(range(10), char_descriptions)
        plt.xlabel('Variance')
        plt.title('Top 10 Most Variable Features')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'feature_variance.png'), dpi=300, bbox_inches='tight')
        plt.close()

        with open(os.path.join(results_dir, 'feature_variance_explanation.txt'), 'w') as f:
            f.write("Top 10 Most Variable Features Explanation\n")
            f.write("=========================================\n\n")
            for i, (char_idx, var) in enumerate(zip(top_features, feature_variance[top_features])):
                desc = CharExplainer.get_char_description(char_idx)
                f.write(f"{i+1}. {desc}: Variance = {var:.6f}\n")

        pca = PCA(n_components=2, random_state=42)
        features_pca = pca.fit_transform(mean_features)
        
        plt.figure(figsize=(10, 8))
        plt.scatter(features_pca[:, 0], features_pca[:, 1], c=unique_labels, cmap='viridis', s=100)
        for i, label in enumerate(unique_labels):
            plt.annotate(f'Class {int(label)}', (features_pca[i, 0], features_pca[i, 1]))
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title('Class Centroids in PCA Space')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(results_dir, 'class_centroids_pca.png'), dpi=300, bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(14, 12))
        if features.shape[1] > 20:
            top_features = np.argsort(feature_variance)[-20:]
            feature_subset = features[:, top_features]
            corr_matrix = np.corrcoef(feature_subset, rowvar=False)
            feature_names_subset = [CharExplainer.get_char_description(i) for i in top_features]
        else:
            corr_matrix = np.corrcoef(features, rowvar=False)
            feature_names_subset = [CharExplainer.get_char_description(i) for i in range(features.shape[1])]
        
        sns.heatmap(corr_matrix, xticklabels=feature_names_subset, yticklabels=feature_names_subset,
                   cmap='coolwarm', vmin=-1, vmax=1, annot=False, fmt='.2f')
        plt.title('Feature Correlation Matrix')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'feature_correlation.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        with open(os.path.join(results_dir, 'feature_correlation_explanation.txt'), 'w') as f:
            f.write("Feature Correlation Explanation\n")
            f.write("===============================\n\n")
            f.write("This heatmap shows the correlation between different character features.\n")
            f.write("Positive values (red) indicate that characters tend to appear together.\n")
            f.write("Negative values (blue) indicate that characters tend to exclude each other.\n\n")

            np.fill_diagonal(corr_matrix, 0)
            max_corr = np.max(corr_matrix)
            min_corr = np.min(corr_matrix)
            
            if max_corr > 0.5:
                max_indices = np.where(corr_matrix == max_corr)
                f.write(f"Strongest positive correlation ({max_corr:.3f}) between:\n")
                for i, j in zip(max_indices[0], max_indices[1]):
                    f.write(f"  - {feature_names_subset[i]} and {feature_names_subset[j]}\n")
            
            if min_corr < -0.5:
                min_indices = np.where(corr_matrix == min_corr)
                f.write(f"Strongest negative correlation ({min_corr:.3f}) between:\n")
                for i, j in zip(min_indices[0], min_indices[1]):
                    f.write(f"  - {feature_names_subset[i]} and {feature_names_subset[j]}\n")

        if len(unique_labels) > 1:
            most_important_feature = np.argmax(feature_variance)
            feature_desc = CharExplainer.get_char_description(most_important_feature)
            
            plt.figure(figsize=(10, 8))
            for label in unique_labels:
                plt.hist(features[labels == label, most_important_feature], 
                         alpha=0.7, label=f'Class {int(label)}', bins=20, density=True)
            plt.xlabel(f'Character Frequency: {feature_desc}')
            plt.ylabel('Density')
            plt.title(f'Distribution of Most Important Feature: {feature_desc}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(results_dir, 'feature_distribution.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        print("Sequence feature analysis plots saved")
    except Exception as e:
        print(f"Error in sequence feature analysis: {str(e)}")

def save_raw_data(smiles, proteins, labels, predictions, results_dir=RESULTS_DIR):
    smiles = np.array(smiles).flatten()
    proteins = np.array(proteins).flatten()
    
    if labels is not None:
        labels = np.array(labels).flatten()
    else:
        labels = ['N/A'] * len(smiles)
    
    predictions = np.array(predictions).flatten()

    min_length = min(len(smiles), len(proteins), len(labels), len(predictions))
    smiles = smiles[:min_length]
    proteins = proteins[:min_length]
    labels = labels[:min_length] if hasattr(labels, '__len__') and not isinstance(labels, str) else [labels] * min_length
    predictions = predictions[:min_length]

    if smiles.ndim > 1:
        smiles = smiles.reshape(-1)
    if proteins.ndim > 1:
        proteins = proteins.reshape(-1)
    if hasattr(labels, 'ndim') and labels.ndim > 1:
        labels = labels.reshape(-1)
    if predictions.ndim > 1:
        predictions = predictions.reshape(-1)
    
    data = {
        'SMILES': smiles,
        'Protein': proteins,
        'True_Label': labels,
        'Prediction': predictions
    }
    
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(results_dir, 'raw_predictions.csv'), index=False)
    print("Raw data saved to raw_predictions.csv")

def main():
    print(f"Using device: {device}")
    
    if TASK == 'validate':
        print("===== Run Verification Mode =====")

        train_data = np.load('processed_data/train.npz')
        train_smiles = train_data['smiles']
        train_proteins = train_data['proteins']

        test_data = np.load('processed_data/test.npz')
        test_smiles = test_data['smiles']
        test_proteins = test_data['proteins']
        test_labels = test_data['labels']
        
        print(f"Test set size: {len(test_smiles)} samples")

        all_smiles_chars = set()
        for s in train_smiles:
            all_smiles_chars.update(s)
        smiles_tokenizer = CharTokenizer(sorted(all_smiles_chars))
        
        all_protein_chars = set()
        for p in train_proteins:
            all_protein_chars.update(p)
        protein_tokenizer = CharTokenizer(sorted(all_protein_chars))

        encoded_smiles = np.array([smiles_tokenizer.encode(s, config.smiles_max_len) for s in test_smiles])
        encoded_proteins = np.array([protein_tokenizer.encode(p, config.protein_max_len) for p in test_proteins])

        encoded_smiles = torch.LongTensor(encoded_smiles)
        encoded_proteins = torch.LongTensor(encoded_proteins)
        test_labels = torch.FloatTensor(test_labels)

        test_dataset = ORLigandDataset(encoded_smiles, encoded_proteins, test_labels)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

        model = ORLigandTransformer(
            config,
            smiles_tokenizer.vocab_size,
            protein_tokenizer.vocab_size
        ).to(device)

        model_path = 'final_model.pth'
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Successfully loaded model weights: {model_path}")

        explainer = SMILESAttentionExplainer(model, device, smiles_tokenizer, protein_tokenizer) if LOCAL_SAMPLE_INDICES else None

        results = evaluate_model(
            model, test_loader, device, "Final Model", 
            explainer=explainer, 
            local_sample_indices=LOCAL_SAMPLE_INDICES,
            smiles_strings=test_smiles,
            protein_strings=test_proteins
        )

        save_raw_data(test_smiles, test_proteins, test_labels, results['probabilities'])

        print("Creating global explanation plots...")
        create_global_explanation_plots(results['probabilities'], results['true_labels'])

        print("Visualizing sequence features...")
        visualize_sequence_features(test_smiles, results['true_labels'])

        print("Analyzing sequence features...")
        analyze_sequence_features(test_smiles, results['true_labels'])
        
        print("\nEvaluation completed! Results saved to results directory")
    
    elif TASK == 'predict':
        print("===== Run Prediction Mode =====")

        train_data = np.load('processed_data/train.npz')
        train_smiles = train_data['smiles']
        train_proteins = train_data['proteins']

        all_smiles_chars = set()
        for s in train_smiles:
            all_smiles_chars.update(s)
        smiles_tokenizer = CharTokenizer(sorted(all_smiles_chars))
        
        all_protein_chars = set()
        for p in train_proteins:
            all_protein_chars.update(p)
        protein_tokenizer = CharTokenizer(sorted(all_protein_chars))

        print(f"Loading prediction data from: {PREDICTION_FILE}")
        try:
            data_lines = []
            with open(PREDICTION_FILE, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        smiles = parts[0].strip()
                        protein = ' '.join(parts[1:]).strip()
                        data_lines.append((smiles, protein))
            
            predict_smiles = [item[0] for item in data_lines]
            predict_proteins = [item[1] for item in data_lines]
            
            print(f"Loaded {len(predict_smiles)} prediction samples")
        except Exception as e:
            print(f"Error loading prediction file: {str(e)}")
            return

        encoded_smiles = []
        for s in predict_smiles:
            try:
                encoded = smiles_tokenizer.encode(s, config.smiles_max_len)
                encoded_smiles.append(encoded)
            except Exception as e:
                print(f"Error encoding SMILES: {s}, error: {str(e)}")
                encoded_smiles.append([0] * config.smiles_max_len)
        
        encoded_proteins = []
        for p in predict_proteins:
            try:
                encoded = protein_tokenizer.encode(p, config.protein_max_len)
                encoded_proteins.append(encoded)
            except Exception as e:
                print(f"Error encoding protein: {p}, error: {str(e)}")
                encoded_proteins.append([0] * config.protein_max_len)

        encoded_smiles = np.array(encoded_smiles)
        encoded_proteins = np.array(encoded_proteins)

        encoded_smiles = torch.LongTensor(encoded_smiles)
        encoded_proteins = torch.LongTensor(encoded_proteins)

        predict_dataset = ORLigandDataset(encoded_smiles, encoded_proteins)
        predict_loader = DataLoader(predict_dataset, batch_size=config.batch_size, shuffle=False)

        model = ORLigandTransformer(
            config,
            smiles_tokenizer.vocab_size,
            protein_tokenizer.vocab_size
        ).to(device)

        model_path = 'final_model.pth'
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Successfully loaded model weights: {model_path}")

        explainer = SMILESAttentionExplainer(model, device, smiles_tokenizer, protein_tokenizer) if LOCAL_SAMPLE_INDICES else None

        predictions = evaluate_model(
            model, predict_loader, device, "Final Model",
            explainer=explainer,
            local_sample_indices=LOCAL_SAMPLE_INDICES,
            smiles_strings=predict_smiles,
            protein_strings=predict_proteins
        )

        print(f"Saving predictions to: {OUTPUT_FILE}")
        try:
            with open(OUTPUT_FILE, 'w') as f:
                f.write("SMILES\tProtein\tPrediction\n")
                for i in range(len(predict_smiles)):
                    smi = str(predict_smiles[i])
                    prot = str(predict_proteins[i])
                    pred = float(predictions[i]) if not isinstance(predictions[i], str) else 0.0
                    
                    f.write(f"{smi}\t{prot}\t{pred:.6f}\n")
        except Exception as e:
            print(f"Error saving predictions: {str(e)}")

        save_raw_data(predict_smiles, predict_proteins, None, predictions)
        
        print(f"Prediction completed! Results saved to: {OUTPUT_FILE}")
        print("Raw data saved to results directory")
    
    else:
        print(f"Error: Unknown task '{TASK}'. Please use 'validate' or 'predict'.")

if __name__ == '__main__':
    main()