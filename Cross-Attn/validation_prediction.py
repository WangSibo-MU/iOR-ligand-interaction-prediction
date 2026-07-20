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
PREDICTION_FILE = 'test.txt'
OUTPUT_FILE = 'predictions.txt'
LOCAL_SAMPLE_INDICES = [0, 1, 2, 3]

# Model output convention. Keep False when ORLigandTransformer already returns
# sigmoid probabilities; set True only when it returns raw logits.
MODEL_OUTPUT_IS_LOGIT = False

# Cross-attention aggregation used for ligand-token importance.
# 'topk_mean' avoids reducing every softmax row to an almost constant 1/L value.
# Set to 'mean' to reproduce the previous visualization behavior.
ATTENTION_REDUCTION = 'topk_mean'  # 'topk_mean', 'max', 'l2', or 'mean'
ATTENTION_TOPK_FRACTION = 0.05
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

def output_to_probability_tensor(output):
    """Convert model output to probabilities using one explicit convention."""
    if not torch.is_tensor(output):
        output = torch.as_tensor(output, dtype=torch.float32, device=device)
    return torch.sigmoid(output) if MODEL_OUTPUT_IS_LOGIT else output


def safe_minmax(values):
    """Return a stable 0-1 normalization without amplifying a constant vector."""
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return arr
    finite = np.isfinite(arr)
    if not finite.all():
        replacement = float(np.nanmedian(arr[finite])) if finite.any() else 0.0
        arr = np.where(finite, arr, replacement)
    vmin = float(arr.min())
    vmax = float(arr.max())
    if np.isclose(vmax, vmin):
        return np.zeros_like(arr)
    return (arr - vmin) / (vmax - vmin)


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
    """Extract and visualize ligand-protein cross-attention robustly.

    The model is not changed. The modifications concern only attention-axis
    handling, padding removal, SMILES-to-atom mapping, and display scaling.
    """

    # Atom tokens in ordinary character-level SMILES. Ring-closure digits and
    # bond symbols are intentionally excluded because they are not atoms.
    ATOM_TOKEN_PATTERN = re.compile(
        r'\[[^\]]+\]|Br|Cl|Si|Se|Na|Li|Mg|Al|Ca|Fe|Zn|Cu|Mn|Hg|Pb|Sn|Ag|Au|'
        r'Pt|Pd|Co|Ni|As|Ba|Bi|Be|Cs|Rb|Sr|Cr|Cd|In|Tl|Sb|Te|Xe|Kr|He|Ne|Ar|'
        r'[A-Z]|[bcnops]'
    )

    def __init__(self, model, device, smiles_tokenizer, protein_tokenizer):
        self.model = model
        self.device = device
        self.smiles_tokenizer = smiles_tokenizer
        self.protein_tokenizer = protein_tokenizer
        self.model.eval()
        self.last_attention_metadata = {}

    @staticmethod
    def _valid_token_count(token_tensor):
        token_tensor = token_tensor.reshape(-1)
        count = int((token_tensor != 0).sum().item())
        return count if count > 0 else int(token_tensor.numel())

    def _orient_attention_matrix(self, attention_weights, smiles_tensor, protein_tensor):
        """Return a [batch, smiles_token, protein_token] matrix.

        Supports common outputs [B,H,Ls,Lp], [B,Ls,Lp], [H,Ls,Lp] for a
        single sample, and transposed ligand/protein axes.
        """
        att = attention_weights.detach()
        batch_size = int(smiles_tensor.shape[0])
        smiles_total = int(smiles_tensor.shape[1])
        protein_total = int(protein_tensor.shape[1])

        if att.dim() == 4:
            # Standard multi-head layout: [B, H, Q, K].
            att = att.mean(dim=1)
        elif att.dim() == 3:
            # Distinguish [B,Q,K] from [H,Q,K] for a one-sample call.
            if att.shape[0] != batch_size and batch_size == 1:
                att = att.mean(dim=0, keepdim=True)
        elif att.dim() == 2:
            att = att.unsqueeze(0)
        else:
            raise ValueError(f"Unsupported attention tensor shape: {tuple(att.shape)}")

        if att.dim() != 3:
            raise ValueError(f"Attention could not be reduced to 3D: {tuple(att.shape)}")

        q_len, k_len = int(att.shape[-2]), int(att.shape[-1])
        direct_error = abs(q_len - smiles_total) + abs(k_len - protein_total)
        reverse_error = abs(q_len - protein_total) + abs(k_len - smiles_total)
        transposed = reverse_error < direct_error
        if transposed:
            att = att.transpose(-2, -1)

        self.last_attention_metadata = {
            'original_shape': tuple(attention_weights.shape),
            'oriented_shape': tuple(att.shape),
            'transposed': transposed,
            'reduction': ATTENTION_REDUCTION,
        }
        return att

    @staticmethod
    def _aggregate_ligand_attention(matrix):
        """Aggregate each ligand row over non-padding protein positions."""
        if matrix.ndim != 2 or matrix.shape[1] == 0:
            return np.zeros(matrix.shape[0] if matrix.ndim else 0, dtype=np.float64)

        matrix = np.asarray(matrix, dtype=np.float64)
        if ATTENTION_REDUCTION == 'mean':
            return matrix.mean(axis=1)
        if ATTENTION_REDUCTION == 'max':
            return matrix.max(axis=1)
        if ATTENTION_REDUCTION == 'l2':
            return np.sqrt(np.mean(np.square(matrix), axis=1))
        if ATTENTION_REDUCTION == 'topk_mean':
            fraction = float(np.clip(ATTENTION_TOPK_FRACTION, 1e-6, 1.0))
            k = max(1, int(np.ceil(matrix.shape[1] * fraction)))
            topk = np.partition(matrix, matrix.shape[1] - k, axis=1)[:, -k:]
            return topk.mean(axis=1)
        raise ValueError(f"Unknown ATTENTION_REDUCTION: {ATTENTION_REDUCTION}")

    def extract_attention(self, smiles_tensor, protein_tensor):
        smiles_tensor = smiles_tensor.clone().detach().to(self.device)
        protein_tensor = protein_tensor.clone().detach().to(self.device)

        with torch.no_grad():
            output, attention_weights = self.model(smiles_tensor, protein_tensor)

        prediction = float(output_to_probability_tensor(output).reshape(-1)[0].item())
        if attention_weights is None:
            return None, None, prediction

        oriented = self._orient_attention_matrix(
            attention_weights, smiles_tensor, protein_tensor
        )
        smiles_valid = min(
            self._valid_token_count(smiles_tensor[0]), int(oriented.shape[-2])
        )
        protein_valid = min(
            self._valid_token_count(protein_tensor[0]), int(oriented.shape[-1])
        )

        matrix = oriented[0, :smiles_valid, :protein_valid].cpu().numpy()
        smiles_attention = self._aggregate_ligand_attention(matrix)
        self.last_attention_metadata.update({
            'valid_smiles_tokens': smiles_valid,
            'valid_protein_tokens': protein_valid,
            'raw_score_min': float(smiles_attention.min()) if smiles_attention.size else 0.0,
            'raw_score_max': float(smiles_attention.max()) if smiles_attention.size else 0.0,
        })
        return smiles_attention, matrix, prediction

    @classmethod
    def _atom_token_spans(cls, smiles_str, mol):
        spans = [(m.start(), m.end(), m.group(0)) for m in cls.ATOM_TOKEN_PATTERN.finditer(smiles_str)]
        if len(spans) != mol.GetNumAtoms():
            raise ValueError(
                f"SMILES atom-token count ({len(spans)}) does not match RDKit atom count "
                f"({mol.GetNumAtoms()}) for {smiles_str!r}."
            )
        return spans

    def map_char_attention_to_atoms(self, smiles_str, attention_weights):
        """Map character-level scores to RDKit atoms without treating ring digits as atoms."""
        mol = Chem.MolFromSmiles(smiles_str)
        if mol is None:
            return None
        try:
            spans = self._atom_token_spans(smiles_str, mol)
            scores = np.asarray(attention_weights, dtype=np.float64).reshape(-1)
            atom_attention = []
            for start, end, _ in spans:
                valid_positions = [j for j in range(start, end) if j < scores.size]
                atom_attention.append(
                    float(scores[valid_positions].mean()) if valid_positions else 0.0
                )
            return atom_attention
        except Exception as exc:
            print(f"Error in mapping character attention to atoms: {exc}")
            return None

    def _save_attention_tables(self, smiles_str, attention_weights, atom_weights, output_path):
        stem = os.path.splitext(output_path)[0]
        char_rows = []
        scores = np.asarray(attention_weights, dtype=np.float64).reshape(-1)
        display_scores = safe_minmax(scores)
        for i, char in enumerate(smiles_str):
            char_rows.append({
                'Char_Index': i,
                'Character': char,
                'Raw_Attention_Score': float(scores[i]) if i < scores.size else 0.0,
                'Relative_Attention_Score': float(display_scores[i]) if i < display_scores.size else 0.0,
            })
        pd.DataFrame(char_rows).to_csv(f"{stem}_character_scores.csv", index=False)

        mol = Chem.MolFromSmiles(smiles_str)
        if mol is not None and atom_weights is not None:
            atom_raw = np.asarray(atom_weights, dtype=np.float64)
            atom_rel = safe_minmax(atom_raw)
            atom_rows = []
            for atom in mol.GetAtoms():
                idx = atom.GetIdx()
                atom_rows.append({
                    'Atom_Index': idx,
                    'Atom_Symbol': atom.GetSymbol(),
                    'Is_Aromatic': atom.GetIsAromatic(),
                    'Raw_Attention_Score': float(atom_raw[idx]),
                    'Relative_Attention_Score': float(atom_rel[idx]),
                })
            pd.DataFrame(atom_rows).to_csv(f"{stem}_atom_scores.csv", index=False)

        with open(f"{stem}_metadata.txt", 'w', encoding='utf-8') as handle:
            for key, value in self.last_attention_metadata.items():
                handle.write(f"{key}: {value}\n")

    def visualize_smiles_attention(self, smiles_str, attention_weights, output_path,
                                    title="SMILES Attention", true_label=None, prediction=None):
        valid_chars = [c for c in smiles_str if not c.isspace() and c != '\x00']
        raw = np.asarray(attention_weights, dtype=np.float64).reshape(-1)[:len(valid_chars)]
        if not valid_chars or raw.size == 0:
            print(f"Warning: No valid character attention for {output_path}")
            return
        relative = safe_minmax(raw)

        full_title = title
        if true_label is not None and prediction is not None:
            full_title = f"{title}\nActual: {int(true_label)}, Predicted probability: {prediction:.4f}"
        elif prediction is not None:
            full_title = f"{title}\nPredicted probability: {prediction:.4f}"

        fig, ax = plt.subplots(figsize=(max(12, len(valid_chars) * 0.8), 6))
        cmap = plt.cm.Reds
        norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
        for i, (char, score) in enumerate(zip(valid_chars, relative)):
            ax.text(
                i, 0.5, char, fontsize=15, ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor=cmap(norm(score)), alpha=0.75)
            )
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Relative attention score (0-1)', fontsize=15)
        ax.set_xlim(-0.5, len(valid_chars) - 0.5)
        ax.set_ylim(0, 1)
        ax.set_title(full_title, fontsize=17)
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"SMILES attention visualization saved to {output_path}")

    def visualize_molecule_attention(self, smiles_str, attention_weights, output_path,
                                     title="Molecular Attention", true_label=None, prediction=None):
        try:
            mol = Chem.MolFromSmiles(smiles_str)
            if mol is None:
                print(f"Warning: Could not parse SMILES: {smiles_str}")
                return

            atom_weights = self.map_char_attention_to_atoms(smiles_str, attention_weights)
            if atom_weights is None:
                print(f"Warning: Atom mapping failed; molecular attention was not drawn for {smiles_str}")
                return

            atom_raw = np.asarray(atom_weights, dtype=np.float64)
            atom_relative = safe_minmax(atom_raw).tolist()
            self._save_attention_tables(smiles_str, attention_weights, atom_raw, output_path)

            full_title = title
            if true_label is not None and prediction is not None:
                full_title = f"{title}\nActual: {int(true_label)}, Predicted probability: {prediction:.4f}"
            elif prediction is not None:
                full_title = f"{title}\nPredicted probability: {prediction:.4f}"

            from rdkit.Chem.Draw import MolDraw2DCairo
            drawer = MolDraw2DCairo(800, 800)
            SimilarityMaps.GetSimilarityMapFromWeights(
                mol,
                atom_relative,
                drawer,
                colorMap=plt.cm.Reds,
                contourLines=10,
                coordScale=1.5,
            )
            drawer.FinishDrawing()
            png_data = drawer.GetDrawingText()
            img = Image.open(io.BytesIO(png_data))

            fig, (ax_img, ax_cbar) = plt.subplots(
                1, 2, figsize=(12, 8), gridspec_kw={'width_ratios': [5, 1]}
            )
            ax_img.imshow(np.array(img))
            ax_img.set_title(full_title, fontsize=19, pad=20)
            ax_img.axis('off')
            norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
            sm = plt.cm.ScalarMappable(cmap=plt.cm.Reds, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, cax=ax_cbar)
            cbar.set_label('Relative attention score (0-1)', fontsize=15)
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Molecular attention visualization saved to {output_path}")
        except Exception as exc:
            print(f"Error in molecular attention visualization: {exc}")
            import traceback
            traceback.print_exc()

    def visualize_cross_attention_heatmap(self, smiles_str, protein_str, cross_attention_matrix,
                                          output_path, title="Cross Attention Heatmap",
                                          true_label=None, prediction=None, aa_per_row=60):
        try:
            matrix = np.asarray(cross_attention_matrix, dtype=np.float64)
            n_smiles = min(len(smiles_str), matrix.shape[0])
            n_protein = min(len(protein_str.rstrip()), matrix.shape[1])
            valid_smiles_chars = list(smiles_str[:n_smiles])
            valid_protein_chars = list(protein_str[:n_protein])
            valid_matrix = matrix[:n_smiles, :n_protein]
            if valid_matrix.size == 0:
                print(f"Warning: No valid attention matrix for {output_path}")
                return

            full_title = title
            if true_label is not None and prediction is not None:
                full_title = f"{title}\nActual: {int(true_label)}, Predicted probability: {prediction:.4f}"
            elif prediction is not None:
                full_title = f"{title}\nPredicted probability: {prediction:.4f}"

            n_rows = max(1, (len(valid_protein_chars) + aa_per_row - 1) // aa_per_row)
            fig, axes = plt.subplots(
                n_rows, 1,
                figsize=(max(12, aa_per_row * 0.5), max(4 * n_rows, 6)),
                squeeze=False,
            )
            axes = axes[:, 0]
            vmin, vmax = float(valid_matrix.min()), float(valid_matrix.max())
            if np.isclose(vmin, vmax):
                vmax = vmin + 1e-12

            for row_idx, ax in enumerate(axes):
                start_idx = row_idx * aa_per_row
                end_idx = min((row_idx + 1) * aa_per_row, len(valid_protein_chars))
                chars = valid_protein_chars[start_idx:end_idx]
                current = valid_matrix[:, start_idx:end_idx]
                im = ax.imshow(
                    current, cmap='Reds', aspect='auto', interpolation='nearest',
                    vmin=vmin, vmax=vmax
                )
                ax.set_xticks(range(len(chars)))
                ax.set_xticklabels(chars, fontsize=9)
                ax.set_yticks(range(len(valid_smiles_chars)))
                ax.set_yticklabels(valid_smiles_chars, fontsize=9)
                ax.set_ylabel('SMILES token', fontsize=10)
                ax.set_title(f"Protein positions {start_idx + 1}-{end_idx}", fontsize=10)

            cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
            cbar = fig.colorbar(im, cax=cbar_ax)
            cbar.set_label('Head-averaged cross-attention weight', fontsize=13)
            fig.suptitle(full_title, fontsize=17, y=0.99)
            fig.text(0.5, 0.015, 'Protein sequence position', ha='center', fontsize=13)
            plt.tight_layout(rect=[0.03, 0.03, 0.9, 0.96])
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()

            matrix_path = output_path.replace('.png', '_matrix.txt')
            np.savetxt(matrix_path, valid_matrix, fmt='%.8f')
            print(f"Cross-attention heatmap saved to {output_path}")
        except Exception as exc:
            print(f"Error in cross-attention visualization: {exc}")
            import traceback
            traceback.print_exc()

    def explain(self, smiles_tensor, protein_tensor, smiles_str, protein_str):
        smiles_attention, cross_attention_matrix, prediction = self.extract_attention(
            smiles_tensor, protein_tensor
        )
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
            probs = output_to_probability_tensor(outputs).detach().cpu().numpy().astype(float)
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

        n_samples = features.shape[0]
        if n_samples < 3:
            raise ValueError(f"Too few samples for t-SNE: n_samples={n_samples}")

        perplexity = max(1, min(30, n_samples - 1))

        # scikit-learn >= 1.5 uses max_iter; older versions used n_iter.
        import inspect
        tsne_params = inspect.signature(TSNE).parameters
        tsne_kwargs = {
            "n_components": 2,
            "random_state": 42,
            "perplexity": perplexity,
        }
        if "max_iter" in tsne_params:
            tsne_kwargs["max_iter"] = 300
        elif "n_iter" in tsne_params:
            tsne_kwargs["n_iter"] = 300
        if "n_jobs" in tsne_params:
            tsne_kwargs["n_jobs"] = 1

        tsne = TSNE(**tsne_kwargs)
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
            n_components = min(2, features.shape[0], features.shape[1])
            if n_components < 2:
                raise ValueError(
                    f"Too few samples/features for 2D PCA: "
                    f"n_samples={features.shape[0]}, n_features={features.shape[1]}"
                )

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

        if mean_features.ndim == 2 and min(mean_features.shape) >= 2 and len(unique_labels) >= 2:
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
        else:
            print(
                "Skipping class centroid PCA: need at least 2 classes and 2 feature dimensions. "
                f"unique_labels={len(unique_labels)}, mean_features_shape={mean_features.shape}"
            )

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

        test_data = np.load('processed_data/external_validation.npz')  ###在这里修改
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
