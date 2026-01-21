import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.explain import Explainer, GNNExplainer
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, recall_score, matthews_corrcoef, roc_curve, confusion_matrix
from sklearn.manifold import TSNE
import warnings
import math
import os
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from rdkit.Chem.Draw import MolDraw2DCairo
from collections import defaultdict
import pickle
import networkx as nx
from matplotlib.colors import LinearSegmentedColormap
from torch_geometric.utils import to_networkx
from PIL import Image, ImageDraw, ImageFont
import io
import pandas as pd
from tqdm import tqdm
import random
from utils import CompoundGNN, ProteinTransformer, CPIPredictor, PositionalEncoding, CharTokenizer, Config
os.environ["LOKY_MAX_CPU_COUNT"] = "4"
warnings.filterwarnings('ignore')

# ===================== User Configuration Area =====================
TASK = 'validate'  # 'validate' or 'predict'
PREDICTION_FILE = 'test.txt'  # Prediction task input file
OUTPUT_FILE = 'predictions.txt'  # Prediction result output file
LOCAL_SAMPLE_INDICES = [608, 609, 610]  # Manually specify the sample index for local interpretation (starting from 0)
GLOBAL_EXPLANATION = True  # Whether to generate a global explanation
# =====================================================

# RANDOM_SEED
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

config = Config()

atom_dict = defaultdict(lambda: len(atom_dict))
fingerprint_dict = defaultdict(lambda: len(fingerprint_dict))
edge_dict = defaultdict(lambda: len(edge_dict))
protein_chars = None

atom_dict_rev = {}
fingerprint_dict_rev = {}
edge_dict_rev = {}

def process_protein_sequence(protein, max_len=480):
    if len(protein) > max_len:
        return protein[:max_len]
    else:
        return protein.ljust(max_len)

def create_atoms(mol):
    atoms = []
    for a in mol.GetAtoms():
        atom_type = a.GetSymbol()
        degree = a.GetDegree()
        formal_charge = a.GetFormalCharge()
        is_aromatic = a.GetIsAromatic()
        hybridization = a.GetHybridization()

        atom_features = (atom_type, degree, formal_charge, is_aromatic, hybridization)
        atoms.append(atom_features)

    atoms = [atom_dict[a] for a in atoms]
    return np.array(atoms)

def create_ijbonddict(mol):
    i_jbond_dict = defaultdict(lambda: [])
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bond = str(b.GetBondType())
        i_jbond_dict[i].append((j, bond))
        i_jbond_dict[j].append((i, bond))
    return i_jbond_dict

def extract_fingerprints(atoms, i_jbond_dict, radius):
    if len(atoms) == 1 or radius == 0:
        fingerprints = [fingerprint_dict[a] for a in atoms]
    else:
        nodes = atoms
        i_jedge_dict = i_jbond_dict
        for _ in range(radius):
            fingerprints = []
            for i, j_edge in i_jedge_dict.items():
                neighbors = [(nodes[j], edge) for j, edge in j_edge]
                fingerprint = (nodes[i], tuple(sorted(neighbors)))
                fingerprints.append(fingerprint_dict[fingerprint])
            nodes = fingerprints
            _i_jedge_dict = defaultdict(lambda: [])
            for i, j_edge in i_jbond_dict.items():
                for j, edge in j_edge:
                    both_side = tuple(sorted((nodes[i], nodes[j])))
                    edge = edge_dict[(both_side, edge)]
                    _i_jedge_dict[i].append((j, edge))
            i_jbond_dict = _i_jedge_dict

    node_features = []
    for atom, fingerprint in zip(atoms, fingerprints):
        node_features.append([float(atom), float(fingerprint)])

    return np.array(node_features, dtype=np.float32)

def create_adjacency(mol, max_nodes=50):
    adjacency = Chem.GetAdjacencyMatrix(mol)
    if adjacency.shape[0] < max_nodes:
        adjacency = np.pad(adjacency, ((0, max_nodes - adjacency.shape[0]), (0, max_nodes - adjacency.shape[1])), mode='constant')
    else:
        adjacency = adjacency[:max_nodes, :max_nodes]
    return np.array(adjacency)

def smiles_to_graph(smiles, radius, max_nodes=50, max_edges=100):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"Invalid SMILES: {smiles}")
        return None, None

    atoms = create_atoms(mol)
    i_jbond_dict = create_ijbonddict(mol)
    fingerprints = extract_fingerprints(atoms, i_jbond_dict, radius)

    if fingerprints.shape[0] < max_nodes:
        pad_width = ((0, max_nodes - fingerprints.shape[0]), (0, 0))
        fingerprints = np.pad(fingerprints, pad_width, mode='constant')
    else:
        fingerprints = fingerprints[:max_nodes]

    edge_index = []
    edge_features = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_index.append((i, j))
        edge_index.append((j, i))

        bond_feature = [
            bond.GetBondTypeAsDouble(), 
            bond.IsInRing(),
            bond.GetIsConjugated(),
            bond.GetStereo(),
        ]
        edge_features.append(bond_feature)
        edge_features.append(bond_feature)

    edge_index = np.array(edge_index, dtype=np.int64).T
    edge_features = np.array(edge_features, dtype=np.float32)

    if edge_index.shape[1] < max_edges:
        pad_width = ((0, 0), (0, max_edges - edge_index.shape[1]))
        edge_index = np.pad(edge_index, pad_width, mode='constant')
        edge_features = np.pad(edge_features, ((0, max_edges - edge_features.shape[0]), (0, 0)), mode='constant')
    else:
        edge_index = edge_index[:, :max_edges]
        edge_features = edge_features[:max_edges]

    x = torch.tensor(fingerprints, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_attr = torch.tensor(edge_features, dtype=torch.float)

    adjacency = create_adjacency(mol, max_nodes=max_nodes)
    adjacency = torch.tensor(adjacency, dtype=torch.float)

    x_with_adjacency = torch.cat([x, adjacency], dim=1)

    adjacency_edge = adjacency[edge_index[0], edge_index[1]].unsqueeze(1)
    edge_attr_with_adjacency = torch.cat([edge_attr, adjacency_edge], dim=1)

    data = Data(x=x_with_adjacency, edge_index=edge_index, edge_attr=edge_attr_with_adjacency)
    return data, mol

class CPIPredictorWrapper(nn.Module):
    def __init__(self, config, protein_vocab_size, node_in_dim):
        super(CPIPredictorWrapper, self).__init__()
        self.model = CPIPredictor(config, protein_vocab_size, node_in_dim)
        self._protein_data = None
        
    def forward(self, x, edge_index, edge_attr=None, batch=None, **kwargs):
        protein_data = kwargs.get('protein_data', self._protein_data)

        if protein_data is None and len(kwargs) == 0 and hasattr(self, '_protein_data'):
            protein_data = self._protein_data

        compound_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        if batch is not None:
            compound_data.batch = batch
            
        return self.model(compound_data, protein_data)

# ============== GNNExplainer ==============
def gnn_explainer_local_explanation(model, test_loader, sample_indices, mol_objects, device, protein_vocab_size, node_in_dim):
    print("\nPerforming GNNExplainer-based local explanation...")
    
    try:
        wrapped_model = CPIPredictorWrapper(
            config=config,
            protein_vocab_size=protein_vocab_size,
            node_in_dim=node_in_dim
        ).to(device)
        
        wrapped_model.model.load_state_dict(model.state_dict())
        wrapped_model.eval()
        
        explainer = Explainer(
            model=wrapped_model,
            algorithm=GNNExplainer(epochs=200),
            explanation_type='model',
            model_config=dict(
                mode='binary_classification',
                task_level='graph',
                return_type='raw',
            ),
            node_mask_type='attributes',
            edge_mask_type='object',
        )
        
        all_data = []
        all_proteins = []
        all_labels = []
        
        for compound_data, protein, labels in test_loader:
            all_data.append(compound_data)
            all_proteins.append(protein)
            all_labels.extend(labels.cpu().numpy())
        
        all_data = Batch.from_data_list([d for batch in all_data for d in batch.to_data_list()])
        all_proteins = torch.cat(all_proteins, dim=0)
        
        for idx in sample_indices:
            print(f"\nAnalyzing sample {idx}...")

            data_sample = all_data.get_example(idx)
            protein_sample = all_proteins[idx:idx+1]
            mol = mol_objects[idx]
            
            if mol is None:
                print(f"Skipping sample {idx} due to invalid molecule")
                continue
            
            actual_label = int(all_labels[idx])

            data_sample = data_sample.to(device)
            protein_sample = protein_sample.to(device)

            with torch.no_grad():
                wrapped_model._protein_data = protein_sample
                pred = wrapped_model(data_sample.x, data_sample.edge_index, 
                                   edge_attr=data_sample.edge_attr, 
                                   batch=data_sample.batch if hasattr(data_sample, 'batch') else None,
                                   protein_data=protein_sample)
                pred_prob = torch.sigmoid(pred).item()

            explanation = explainer(
                x=data_sample.x,
                edge_index=data_sample.edge_index,
                edge_attr=data_sample.edge_attr,
                batch=data_sample.batch if hasattr(data_sample, 'batch') else None,
                target=None
            )
            
            node_mask = explanation.node_mask
            edge_mask = explanation.edge_mask

            visualize_molecule_explanation(mol, node_mask, edge_mask, idx, actual_label, pred_prob)

            save_feature_importance_csv(mol, node_mask, edge_mask, idx, actual_label, pred_prob)
            
            print(f"GNNExplainer explanation for sample {idx} saved.")
        
        print("\nGNNExplainer-based local explanation completed.")
        
    except Exception as e:
        print(f"Error in GNNExplainer-based local explanation: {str(e)}")
        import traceback
        traceback.print_exc()

def save_feature_importance_csv(mol, node_mask, edge_mask, sample_idx, actual_label, pred_prob):
    try:
        node_importance = node_mask.cpu().numpy()
        if node_importance.size == 0:
            print(f"No node importance values for sample {sample_idx}")
            return

        if node_importance.ndim > 1:
            node_importance = node_importance.flatten()

        node_importance = (node_importance - node_importance.min()) / (node_importance.max() - node_importance.min() + 1e-8)

        atom_data = []
        for i, atom in enumerate(mol.GetAtoms()):
            if i < len(node_importance):
                importance = float(node_importance[i])
                atom_data.append({
                    'Atom_Index': i,
                    'Atom_Type': atom.GetSymbol(),
                    'Degree': atom.GetDegree(),
                    'Formal_Charge': atom.GetFormalCharge(),
                    'Is_Aromatic': atom.GetIsAromatic(),
                    'Hybridization': str(atom.GetHybridization()),
                    'Importance_Score': importance
                })
        
        atom_df = pd.DataFrame(atom_data)
        atom_csv_path = os.path.join(RESULTS_DIR, f'sample_{sample_idx}_atom_importance.csv')
        atom_df.to_csv(atom_csv_path, index=False)
        print(f"Atom feature importance saved to {atom_csv_path}")
        edge_importance = edge_mask.cpu().numpy()
        if edge_importance.size > 0:
            if edge_importance.ndim > 1:
                edge_importance = edge_importance.flatten()
            edge_importance = (edge_importance - edge_importance.min()) / (edge_importance.max() - edge_importance.min() + 1e-8)
            bond_data = []
            for i, bond in enumerate(mol.GetBonds()):
                if i < len(edge_importance):
                    importance = float(edge_importance[i])
                    begin_atom = bond.GetBeginAtom()
                    end_atom = bond.GetEndAtom()
                    
                    bond_data.append({
                        'Bond_Index': i,
                        'Begin_Atom_Index': bond.GetBeginAtomIdx(),
                        'Begin_Atom_Type': begin_atom.GetSymbol(),
                        'End_Atom_Index': bond.GetEndAtomIdx(),
                        'End_Atom_Type': end_atom.GetSymbol(),
                        'Bond_Type': str(bond.GetBondType()),
                        'Is_In_Ring': bond.IsInRing(),
                        'Is_Conjugated': bond.GetIsConjugated(),
                        'Stereo': str(bond.GetStereo()),
                        'Importance_Score': importance
                    })
            bond_df = pd.DataFrame(bond_data)
            bond_csv_path = os.path.join(RESULTS_DIR, f'sample_{sample_idx}_bond_importance.csv')
            bond_df.to_csv(bond_csv_path, index=False)
            print(f"Bond feature importance saved to {bond_csv_path}")
            
    except Exception as e:
        print(f"Error saving feature importance to CSV: {str(e)}")
        import traceback
        traceback.print_exc()

def visualize_molecule_explanation(mol, node_mask, edge_mask, sample_idx, actual_label, pred_prob):
    try:
        node_importance = node_mask.cpu().numpy()
        if node_importance.size == 0:
            print(f"No node importance values for sample {sample_idx}")
            return
        if node_importance.ndim > 1:
            node_importance = node_importance.flatten()
            
        node_importance = (node_importance - node_importance.min()) / (node_importance.max() - node_importance.min() + 1e-8)
        cmap = plt.get_cmap('Reds')
        atom_colors = {}
        for i, importance in enumerate(node_importance):
            if i < len(node_importance):
                importance_float = float(importance)
                atom_colors[i] = tuple(float(c) for c in cmap(importance_float))
        
        # Drawing molecules
        drawer = MolDraw2DCairo(1000, 1000)
        drawer.SetFontSize(12)  # Font size
        highlight_atoms = []
        highlight_colors = {}
        for i, atom in enumerate(mol.GetAtoms()):
            if i < len(node_importance):
                highlight_atoms.append(i)
                highlight_colors[i] = atom_colors[i]
        highlight_bonds = []
        bond_colors = {}
        edge_importance = edge_mask.cpu().numpy()
        if edge_importance.size > 0:
            if edge_importance.ndim > 1:
                edge_importance = edge_importance.flatten()
                
            edge_importance = (edge_importance - edge_importance.min()) / (edge_importance.max() - edge_importance.min() + 1e-8)
            
            for i, bond in enumerate(mol.GetBonds()):
                if i < len(edge_importance):
                    highlight_bonds.append(i)
                    edge_importance_float = float(edge_importance[i])
                    bond_colors[i] = tuple(float(c) for c in cmap(edge_importance_float))
        drawer.DrawMolecule(
            mol,
            highlightAtoms=highlight_atoms,
            highlightAtomColors=highlight_colors,
            highlightBonds=highlight_bonds,
            highlightBondColors=bond_colors,
        )
        
        drawer.FinishDrawing()
        
        png_data = drawer.GetDrawingText()
        
        img = Image.open(io.BytesIO(png_data))
        
        fig, (ax_img, ax_cbar) = plt.subplots(1, 2, figsize=(12, 10), 
                                             gridspec_kw={'width_ratios': [5, 1]})
        
        ax_img.imshow(img)
        ax_img.axis('off')
        ax_img.set_title(f'Sample {sample_idx}\nActual: {actual_label}, Predicted: {pred_prob:.4f}', fontsize=17)
        
        norm = plt.Normalize(0, 1)
        sm = plt.cm.ScalarMappable(cmap='Reds', norm=norm)
        sm.set_array([])
        
        cbar = plt.colorbar(sm, cax=ax_cbar)
        cbar.set_label('Importance Score', fontsize=15)
        cbar.ax.tick_params(labelsize=10)
        
        plt.tight_layout()
        
        plt.savefig(os.path.join(RESULTS_DIR, f'gnnexplainer_sample_{sample_idx}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
            
    except Exception as e:
        print(f"Error visualizing molecule explanation: {str(e)}")
        import traceback
        traceback.print_exc()

# ============== Global explanation ==============
def create_global_explanation_plots(predictions, labels):
    plt.figure(figsize=(10, 8))
    plt.hist(predictions, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel('Prediction Value')
    plt.ylabel('Frequency')
    plt.title('Distribution of Prediction Values')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(RESULTS_DIR, 'prediction_distribution.png'), dpi=300, bbox_inches='tight')
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
        plt.savefig(os.path.join(RESULTS_DIR, 'prediction_by_class.png'), dpi=300, bbox_inches='tight')
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
        plt.savefig(os.path.join(RESULTS_DIR, 'roc_curve_global.png'), dpi=300, bbox_inches='tight')
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
        ax.set_title('Performance Metrics', fontsize=14)
        
        plt.savefig(os.path.join(RESULTS_DIR, 'performance_metrics.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    print("Global explanation plots saved to results directory")

# ============== Compound feature extraction ==============
def extract_compound_features(model, data_loader, device):
    model.eval()
    compound_features = []
    compound_labels = []
    
    with torch.no_grad():
        for compound_data, protein, labels in data_loader:
            compound_data = compound_data.to(device)
            protein = protein.to(device)
            compound_emb = model.compound_gnn(compound_data)
            if compound_emb.size(0) != compound_data.batch.size(0):
                num_nodes = compound_emb.size(0)
                num_graphs = compound_data.num_graphs
                nodes_per_graph = num_nodes // num_graphs
                batch = torch.repeat_interleave(
                    torch.arange(num_graphs, device=device), 
                    nodes_per_graph
                )
                if num_nodes % num_graphs != 0:
                    extra_nodes = num_nodes - num_graphs * nodes_per_graph
                    batch = torch.cat([batch, torch.full((extra_nodes,), num_graphs-1, device=device)])
            else:
                batch = compound_data.batch
            
            compound_pooled = global_mean_pool(compound_emb, batch)

            compound_features.append(compound_pooled.cpu().detach().numpy())
            compound_labels.extend(labels.cpu().numpy())
    
    compound_features = np.vstack(compound_features)
    compound_labels = np.array(compound_labels)
    
    return compound_features, compound_labels

def visualize_compound_features(features, labels):
    """t-SNE"""
    try:
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=300, n_jobs=1)
        features_2d = tsne.fit_transform(features)

        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='viridis', alpha=0.7)
        plt.colorbar(scatter, label='Class Label')
        plt.title('t-SNE Visualization of Compound Features')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(RESULTS_DIR, 'compound_tsne.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("Compound feature visualization saved")
    except Exception as e:
        print(f"Error in t-SNE visualization: {str(e)}")
        try:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2, random_state=42)
            features_2d = pca.fit_transform(features)
            
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='viridis', alpha=0.7)
            plt.colorbar(scatter, label='Class Label')
            plt.title('PCA Visualization of Compound Features (t-SNE failed)')
            plt.xlabel('Principal Component 1')
            plt.ylabel('Principal Component 2')
            plt.grid(True, alpha=0.3)
            
            plt.savefig(os.path.join(RESULTS_DIR, 'compound_pca.png'), dpi=300, bbox_inches='tight')
            plt.close()
            print("PCA visualization saved as fallback")
        except Exception as e2:
            print(f"PCA visualization also failed: {str(e2)}")

def analyze_compound_features(features, labels):
    try:
        unique_labels = np.unique(labels)
        mean_features = []
        
        for label in unique_labels:
            mean_feature = np.mean(features[labels == label], axis=0)
            mean_features.append(mean_feature)
        
        mean_features = np.array(mean_features)
        feature_variance = np.var(features, axis=0)
        top_features = np.argsort(feature_variance)[-10:]  # Select the top 10 features with the highest variance
        feature_names = []
        for feat_idx in top_features:
            if feat_idx < len(atom_dict_rev):
                feature_names.append(f"Atom: {atom_dict_rev.get(feat_idx, f'Unknown_{feat_idx}')}")
            else:
                fingerprint_idx = feat_idx - len(atom_dict_rev)
                feature_names.append(f"Fingerprint: {fingerprint_dict_rev.get(fingerprint_idx, f'Unknown_{fingerprint_idx}')}")
        
        plt.figure(figsize=(12, 8))
        plt.barh(range(10), feature_variance[top_features])
        plt.yticks(range(10), feature_names, fontsize=8)
        plt.xlabel('Variance')
        plt.title('Top 10 Most Variable Features')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'feature_variance.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        from sklearn.decomposition import PCA
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
        plt.savefig(os.path.join(RESULTS_DIR, 'class_centroids_pca.png'), dpi=300, bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(15, 12))
        if features.shape[1] > 20:
            top_features = np.argsort(feature_variance)[-20:]
            feature_subset = features[:, top_features]
            corr_matrix = np.corrcoef(feature_subset, rowvar=False)
            heatmap_feature_names = []
            for feat_idx in top_features:
                if feat_idx < len(atom_dict_rev):
                    heatmap_feature_names.append(f"A{feat_idx}:{str(atom_dict_rev.get(feat_idx, 'Unknown'))[:20]}")
                else:
                    fingerprint_idx = feat_idx - len(atom_dict_rev)
                    heatmap_feature_names.append(f"F{fingerprint_idx}:{str(fingerprint_dict_rev.get(fingerprint_idx, 'Unknown'))[:20]}")
        else:
            corr_matrix = np.corrcoef(features, rowvar=False)
            heatmap_feature_names = [f"Feature {i}" for i in range(features.shape[1])]
        
        sns.heatmap(corr_matrix, 
                   xticklabels=heatmap_feature_names,
                   yticklabels=heatmap_feature_names,
                   cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('Feature Correlation Matrix')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'feature_correlation.png'), dpi=300, bbox_inches='tight')
        plt.close()

        if len(unique_labels) > 1:
            most_important_feature = np.argmax(feature_variance)
            feature_name = f"Feature {most_important_feature}"
            if most_important_feature < len(atom_dict_rev):
                feature_name = f"Atom: {atom_dict_rev.get(most_important_feature, 'Unknown')}"
            else:
                fingerprint_idx = most_important_feature - len(atom_dict_rev)
                feature_name = f"Fingerprint: {fingerprint_dict_rev.get(fingerprint_idx, 'Unknown')}"
            
            plt.figure(figsize=(10, 8))
            for label in unique_labels:
                plt.hist(features[labels == label, most_important_feature], 
                         alpha=0.7, label=f'Class {int(label)}', bins=20, density=True)
            plt.xlabel(f'{feature_name} Value')
            plt.ylabel('Density')
            plt.title(f'Distribution of Most Important Feature')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(RESULTS_DIR, 'feature_distribution.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        print("Compound feature analysis plots saved")

        with open(os.path.join(RESULTS_DIR, 'feature_mapping.csv'), 'w') as f:
            f.write("Feature_Index,Feature_Type,Feature_Description\n")
            for i in range(min(100, features.shape[1])):  # Save the first 100 features
                if i < len(atom_dict_rev):
                    f.write(f"{i},Atom,{atom_dict_rev.get(i, 'Unknown')}\n")
                else:
                    fingerprint_idx = i - len(atom_dict_rev)
                    f.write(f"{i},Fingerprint,{fingerprint_dict_rev.get(fingerprint_idx, 'Unknown')}\n")
        
    except Exception as e:
        print(f"Error in compound feature analysis: {str(e)}")
        import traceback
        traceback.print_exc()

def save_raw_data(smiles, proteins, labels, predictions):
    data = {
        'SMILES': smiles,
        'Protein': proteins,
        'True_Label': labels if labels is not None else ['N/A'] * len(smiles),
        'Prediction': predictions
    }
    
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(RESULTS_DIR, 'raw_predictions.csv'), index=False)
    print("Raw data saved to raw_predictions.csv")

class ORLigandDataset(Dataset):
    def __init__(self, compound_graphs, proteins, labels, protein_tokenizer, protein_max_len):
        self.compound_graphs = compound_graphs
        self.proteins = proteins
        self.labels = labels
        self.protein_tokenizer = protein_tokenizer
        self.protein_max_len = protein_max_len
        
    def __len__(self):
        return len(self.labels) if self.labels is not None else len(self.compound_graphs)
    
    def __getitem__(self, idx):
        graph = self.compound_graphs[idx]
        
        protein_seq = self.proteins[idx]
        protein_encoded = self.protein_tokenizer.encode(protein_seq, self.protein_max_len)

        label = self.labels[idx] if self.labels is not None else 0
        
        return graph, protein_encoded, label

def collate_fn(batch):
    graphs = [item[0] for item in batch]
    proteins = torch.tensor([item[1] for item in batch], dtype=torch.long)
    labels = torch.tensor([item[2] for item in batch], dtype=torch.float)

    batch_graph = Batch.from_data_list(graphs)
    
    return batch_graph, proteins, labels

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
    
    if len(np.unique(true_labels)) > 1:
        cm = confusion_matrix(true_labels, binary_preds)
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
        f.write(f"Test Set Size: {len(true_labels)}\n")
        f.write(f"AUC: {auc:.4f}\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"MCC: {mcc:.4f}\n")
        f.write(f"Threshold: {optimal_threshold:.4f} (Fixed at 0.5)\n")
    
    return auc, acc, f1, recall, mcc, predictions, true_labels

def adjust_embedding_layer(model, state_dict, expected_vocab_size, embed_dim):
    embedding_key = 'protein_transformer.protein_embedding.weight'
    if embedding_key in state_dict:
        if state_dict[embedding_key].size(0) != expected_vocab_size:
            print(f"Adjust embedding layer size: original size {state_dict[embedding_key].size(0)} -> new size {expected_vocab_size}")

            new_embedding = nn.Embedding(
                expected_vocab_size,
                embed_dim,
                padding_idx=0
            )

            min_size = min(state_dict[embedding_key].size(0), expected_vocab_size)
            new_embedding.weight.data[:min_size] = state_dict[embedding_key][:min_size]

            state_dict[embedding_key] = new_embedding.weight.data
    return state_dict


def load_feature_dicts():
    global atom_dict, fingerprint_dict, edge_dict, protein_chars, atom_dict_rev, fingerprint_dict_rev, edge_dict_rev
    
    try:
        with open('processed_data/atom_dict.pkl', 'rb') as f:
            atom_dict.update(pickle.load(f))
        with open('processed_data/fingerprint_dict.pkl', 'rb') as f:
            fingerprint_dict.update(pickle.load(f))
        with open('processed_data/edge_dict.pkl', 'rb') as f:
            edge_dict.update(pickle.load(f))
        
        atom_dict_rev = {v: k for k, v in atom_dict.items()}
        fingerprint_dict_rev = {v: k for k, v in fingerprint_dict.items()}
        edge_dict_rev = {v: k for k, v in edge_dict.items()}
        
        with open('processed_data/protein_chars.pkl', 'rb') as f:
            global protein_chars
            protein_chars = pickle.load(f)
        
        print("Successfully loaded feature dictionary and protein character set")
        print(f"Atom dictionary size: {len(atom_dict)}")
        print(f"Fingerprint dictionary size: {len(fingerprint_dict)}")
        print(f"Edge dictionary size: {len(edge_dict)}")

        with open(os.path.join(RESULTS_DIR, 'feature_dictionary_details.txt'), 'w') as f:
            f.write("=== Atom Dictionary ===\n")
            for idx, feature in atom_dict_rev.items():
                f.write(f"{idx}: {feature}\n")
            
            f.write("\n=== Fingerprint Dictionary (First 20 entries) ===\n")
            for idx, feature in list(fingerprint_dict_rev.items())[:20]:
                f.write(f"{idx}: {feature}\n")
            
            f.write("\n=== Edge Dictionary ===\n")
            for idx, feature in edge_dict_rev.items():
                f.write(f"{idx}: {feature}\n")
                
        return True
    except Exception as e:
        print(f"Failed to load feature dictionary or protein character set: {str(e)}")
        return False

def main():

    os.makedirs(RESULTS_DIR, exist_ok=True)
    if not load_feature_dicts():
        print("Unable to load feature dictionary, exit")
        return

    protein_tokenizer = CharTokenizer(protein_chars)
    protein_vocab_size = protein_tokenizer.vocab_size
    print(f"Size of protein vocabulary list: {protein_vocab_size}")

    mol_objects = []
    
    if TASK == 'validate':
        print("Validation mode: Load raw test data from test.txt...")
        test_smiles = []
        test_proteins = []
        test_labels = []
        
        try:
            with open('processed_data/test.txt', 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) < 3:
                        continue
                    test_smiles.append(parts[0])
                    test_proteins.append(parts[1])
                    test_labels.append(int(parts[2]))
            
            print(f"Loading {len(test_smiles)} samples")
        except Exception as e:
            print(f"Error loading test data: {str(e)}")
            return

        test_compounds = []
        valid_indices = []
        for idx, smiles in enumerate(test_smiles):
            graph, mol = smiles_to_graph(smiles, radius=2, max_nodes=50, max_edges=100)
            if graph is None:
                print(f"Invalid SMILES: {smiles}")
                continue
            test_compounds.append(graph)
            mol_objects.append(mol)
            valid_indices.append(idx)

        valid_smiles = [test_smiles[i] for i in valid_indices]
        valid_proteins = [test_proteins[i] for i in valid_indices]
        valid_labels = [test_labels[i] for i in valid_indices]

        processed_proteins = [process_protein_sequence(p, config.protein_max_len) for p in valid_proteins]
        
        print(f"Number of effective test samples: {len(test_compounds)}")

        test_dataset = ORLigandDataset(
            test_compounds, processed_proteins, valid_labels,
            protein_tokenizer, config.protein_max_len
        )

        test_loader = DataLoader(
            test_dataset, 
            batch_size=config.batch_size, 
            shuffle=False,
            collate_fn=collate_fn
        )

        node_in_dim = test_compounds[0].x.shape[1] if test_compounds else 0

        model = CPIPredictor(
            config, 
            protein_vocab_size=protein_vocab_size,
            node_in_dim=node_in_dim
        ).to(device)

        try:
            state_dict = torch.load('final_model.pth', map_location=device)

            state_dict = adjust_embedding_layer(model, state_dict, protein_vocab_size, config.protein_embed_dim)
            
            model.load_state_dict(state_dict)
            model.eval()
            print("Model loaded successfully")
            print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return

        print("Model evaluating ...")
        auc, acc, f1, recall, mcc, all_predictions, all_labels = evaluate_model(model, test_loader, device)
        
        print("\nEvaluation results:")
        print(f"AUC: {auc:.4f}")
        print(f"Accuracy: {acc:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"Matthews Correlation Coefficient (MCC): {mcc:.4f}")
        print("ROC curve and confusion matrix have been saved")

        save_raw_data(valid_smiles, valid_proteins, all_labels, all_predictions)

        if GLOBAL_EXPLANATION:
            print("Generating global explanation...")
            create_global_explanation_plots(all_predictions, all_labels)

            print("Extracting compound features for global analysis...")
            compound_features, compound_labels = extract_compound_features(model, test_loader, device)

            print("Creating t-SNE visualization of compound features...")
            visualize_compound_features(compound_features, compound_labels)

            print("Analyzing compound features...")
            analyze_compound_features(compound_features, compound_labels)

        if LOCAL_SAMPLE_INDICES:
            print(f"Explaining samples {LOCAL_SAMPLE_INDICES}...")

            gnn_explainer_local_explanation(
                model, 
                test_loader, 
                LOCAL_SAMPLE_INDICES, 
                mol_objects,
                device,
                protein_vocab_size,
                node_in_dim
            )
            
            print(f"Sample explanations saved")
    
    else:
        print("Prediction mode: Load prediction data...")
        smiles_list = []
        protein_list = []
        
        try:
            with open(PREDICTION_FILE, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        smiles_list.append(parts[0])
                        protein_list.append(' '.join(parts[1:]))
            print(f"Load {len(smiles_list)} samples")
        except Exception as e:
            print(f"Error loading prediction file: {str(e)}")
            return

        compound_graphs = []
        valid_indices = []
        
        for idx, (smiles, protein) in enumerate(zip(smiles_list, protein_list)):
            graph, mol = smiles_to_graph(smiles, radius=2, max_nodes=50, max_edges=100)
            if graph is None:
                print(f"Invalid SMILES: {smiles}")
                continue
            compound_graphs.append(graph)
            mol_objects.append(mol)
            valid_indices.append(idx)

        valid_smiles = [smiles_list[i] for i in valid_indices]
        valid_proteins = [protein_list[i] for i in valid_indices]

        predict_dataset = ORLigandDataset(
            compound_graphs, valid_proteins, None,
            protein_tokenizer, config.protein_max_len
        )

        predict_loader = DataLoader(
            predict_dataset, 
            batch_size=config.batch_size, 
            shuffle=False,
            collate_fn=collate_fn
        )

        if compound_graphs:
            node_in_dim = compound_graphs[0].x.shape[1]
        else:
            print("No valid compound data, exit")
            return

        model = CPIPredictor(
            config, 
            protein_vocab_size=protein_vocab_size,
            node_in_dim=node_in_dim
        ).to(device)

        try:
            state_dict = torch.load('final_model.pth', map_location=device)

            state_dict = adjust_embedding_layer(model, state_dict, protein_vocab_size, config.protein_embed_dim)
            
            model.load_state_dict(state_dict)
            model.eval()
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return

        predictions = []
        with torch.no_grad():
            for batch in predict_loader:
                compound_data, protein, _ = batch
                compound_data = compound_data.to(device)
                protein = protein.to(device)
                outputs = model(compound_data, protein)
                

                if outputs.dim() == 0:
                    batch_preds = [outputs.item()]
                else:
                    batch_preds = outputs.cpu().numpy().tolist()
                
                predictions.extend(batch_preds)

        try:
            with open(OUTPUT_FILE, 'w') as f:
                f.write("SMILES\tProtein\tPrediction\n")
                for i in range(len(valid_smiles)):
                    f.write(f"{valid_smiles[i]}\t{valid_proteins[i]}\t{predictions[i]:.6f}\n")
            print(f"Saving predictions to: {OUTPUT_FILE}")
        except Exception as e:
            print(f"Error saving predictions: {str(e)}")
            return

        save_raw_data(valid_smiles, valid_proteins, None, predictions)

        invalid_indices = set(range(len(smiles_list))) - set(valid_indices)
        if invalid_indices:
            try:
                with open('invalid_samples.txt', 'w') as f:
                    f.write("Invalid SMILES:\n")
                    for i in invalid_indices:
                        f.write(f"{smiles_list[i]}\t{protein_list[i]}\n")
                print(f"Saving invalid SMILES to: invalid_samples.txt")
            except Exception as e:
                print(f"Error saving invalid sample: {str(e)}")
        
        print(f"Prediction completed! Results saved to: {OUTPUT_FILE}")

if __name__ == '__main__':
    main()