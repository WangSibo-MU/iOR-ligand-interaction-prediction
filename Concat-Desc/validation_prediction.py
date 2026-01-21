import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, accuracy_score, f1_score, recall_score,
                             matthews_corrcoef, roc_curve, confusion_matrix)
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import joblib
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
from utils import Config, CharTokenizer, CPIDataset, CPIPredictor
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
import csv
import json

warnings.filterwarnings('ignore')
os.environ['PYTHONHASHSEED'] = '42'
np.random.seed(42)
torch.manual_seed(42)
os.environ['LOKY_MAX_CPU_COUNT'] = '4'

# =====================================================
TASK = 'validate'  # 'validate' or 'predict'
PREDICTION_FILE = 'test.txt'
OUTPUT_FILE = 'predictions.txt'
# =====================================================

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using: {device}")
config = Config()

def evaluate_model(model, data_loader, device, model_name="Final Model"):
    model.eval()
    predictions = []
    true_labels = []
    probabilities = []

    with torch.no_grad():
        for batch in data_loader:
            if len(batch) == 3:
                protein, ligand_features, labels = batch
                protein = protein.to(device)
                ligand_features = ligand_features.float().to(device)
                true_labels.extend(labels.cpu().numpy().tolist())
            else:
                protein, ligand_features = batch
                protein = protein.to(device)
                ligand_features = ligand_features.float().to(device)

            outputs = model(protein, ligand_features) 
            probs = outputs.cpu().numpy().astype(float)
            probabilities.extend(probs.tolist())

            batch_preds = (np.array(probs) > 0.5).astype(int)
            predictions.extend(batch_preds.tolist())

    probabilities = np.asarray(probabilities, dtype=float)
    predictions = np.asarray(predictions, dtype=int)

    if len(true_labels) > 0:
        true_labels = np.asarray(true_labels, dtype=int)

        auc = roc_auc_score(true_labels, probabilities) if len(np.unique(true_labels)) > 1 else 0.5
        acc = accuracy_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions) if len(np.unique(true_labels)) > 1 else 0.0
        recall = recall_score(true_labels, predictions) if np.sum(true_labels) > 0 else 0.0
        mcc = matthews_corrcoef(true_labels, predictions) if len(np.unique(true_labels)) > 1 else 0.0
        fixed_threshold = 0.5 

        cm = confusion_matrix(true_labels, predictions, labels=[0, 1])
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
            f.write(f"Threshold: {fixed_threshold:.4f}\n")

        return {
            'auc': auc,
            'accuracy': acc,
            'f1': f1,
            'recall': recall,
            'mcc': mcc,
            'probabilities': probabilities,
            'true_labels': true_labels,
            'Threshold': fixed_threshold
        }
    else:
        return probabilities

def compute_descriptors_for_prediction(smiles_list, metadata_path='descriptors/descriptor_metadata.json'):
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    selected_names = metadata['selected_names']
    print(f"Calculating {len(selected_names)} descriptors")
    
    calculator = MoleculeDescriptors.MolecularDescriptorCalculator(selected_names)
    
    descriptors = []
    valid_indices = []
    for idx, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSMILES(smiles)
        if mol is not None:
            try:
                desc_values = calculator.CalcDescriptors(mol)
                descriptors.append(desc_values)
                valid_indices.append(idx)
            except Exception as e:
                print(f"Error calculating descriptor for {smiles}: {str(e)}")
                continue
        else:
            print(f"Invalid SMILES: {smiles}")

    descriptors = np.array(descriptors)
    return descriptors, valid_indices

def save_raw_data(smiles, proteins, labels, probabilities, output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame({
        'SMILES': smiles,
        'Protein': proteins,
        'Probability': probabilities.flatten()
    })
    if labels is not None:
        df['True_Label'] = labels
    df.to_csv(os.path.join(output_dir, 'raw_predictions.csv'), index=False)
    print(f"The raw prediction data has been saved to {output_dir}/raw_predictions.csv")

def create_global_explanation_plots(probabilities, true_labels, output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)
    
    fpr, tpr, _ = roc_curve(true_labels, probabilities)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc_score(true_labels, probabilities):.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(8, 6))
    sns.histplot(probabilities, bins=50, kde=True)
    plt.xlabel('Predicted Probability')
    plt.title('Prediction Probability Distribution')
    plt.savefig(os.path.join(output_dir, 'prediction_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Charts have been saved to {output_dir}/")

def visualize_compound_features(descriptors, labels, output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)
    
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(descriptors)
    
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], c=labels, cmap='coolwarm', alpha=0.6)
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    plt.title('PCA Visualization of Compounds')
    plt.colorbar(scatter, label='Activity')
    plt.savefig(os.path.join(output_dir, 'pca_visualization.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Compound feature visualization has been saved to {output_dir}/")

def analyze_compound_features(descriptors, labels, descriptor_names, output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)
    
    correlations = []
    for i, name in enumerate(descriptor_names):
        corr = np.corrcoef(descriptors[:, i], labels)[0, 1]
        correlations.append((name, abs(corr) if not np.isnan(corr) else 0))
    
    correlations.sort(key=lambda x: x[1], reverse=True)
    
    with open(os.path.join(output_dir, 'top_correlated_features.txt'), 'w') as f:
        f.write("Top 20 Most Correlated Features with Activity:\n")
        for name, corr in correlations[:20]:
            f.write(f"{name}: {corr:.4f}\n")
    print(f"Feature analysis results have been saved to {output_dir}/top_correlated_features.txt")

def save_feature_mappings(descriptor_names, output_dir="models"):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'descriptor_name_mapping.json'), 'w') as f:
        mapping = {i: name for i, name in enumerate(descriptor_names)}
        json.dump(mapping, f, indent=2)
    print(f"Feature mapping has been saved to {output_dir}/descriptor_name_mapping.json")

def main():
    try:
        with open('models/protein_tokenizer_chars.txt', 'r') as f:
            protein_chars = f.read().strip().split(',')
        protein_tokenizer = CharTokenizer(protein_chars)
        print(f"Loading protein tokenizer, including {len(protein_chars)} tokens")
    except FileNotFoundError:
        print("Error: Protein tokenizer character file was not found")
        return

    try:
        scaler = joblib.load('models/ligand_scaler.pkl')
        print("Loading the molecular descriptor standardizer")
    except FileNotFoundError:
        print("Error: Molecular descriptor standardizer file was not found")
        return

    vif_names_path = 'descriptors/selected_descriptor_names_vif.txt'
    if os.path.exists(vif_names_path):
        with open(vif_names_path, 'r') as f:
            descriptor_names = f.read().strip().split(',')
        print(f"Loading {len(descriptor_names)} de redundant descriptors")
    else:
        raise FileNotFoundError(
            "Error: The list of de redundant descriptors was not found"
        )

    if TASK == 'validate':
        print("\n" + "="*60)
        print("Verification mode")
        print("="*60)

        print("Loading test data...")
        try:
            test_data = np.load('processed_data/test.npz', allow_pickle=True)
            test_smiles = test_data['smiles']
            test_proteins = test_data['proteins']
            test_labels = test_data['labels']
            test_descriptors = test_data['descriptors']
        except Exception as e:
            print(f"Failed to load test data: {str(e)}")
            return

        print(f"Test set: {len(test_smiles)} samples")

        test_descriptors = scaler.transform(test_descriptors)

        encoded_proteins = np.array([protein_tokenizer.encode(p, config.protein_max_len) for p in test_proteins])
        encoded_proteins_tensor = torch.LongTensor(encoded_proteins)
        ligand_features_tensor = torch.FloatTensor(test_descriptors)
        test_labels_tensor = torch.FloatTensor(test_labels)

        test_dataset = CPIDataset(encoded_proteins_tensor, ligand_features_tensor, test_labels_tensor)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

        print("Loading nn...")
        try:
            model = CPIPredictor(config, protein_tokenizer.vocab_size, ligand_features_tensor.shape[1]).to(device)
            model.load_state_dict(torch.load('models/final_nn_model.pth', map_location=device))
            model.eval()
        except Exception as e:
            print(f"Failed to load the model: {str(e)}")
            return

        print("\nStart evaluating...")
        results = evaluate_model(model, test_loader, device)

        save_raw_data(test_smiles, test_proteins, test_labels, results['probabilities'])
        create_global_explanation_plots(results['probabilities'], results['true_labels'])
        visualize_compound_features(test_descriptors, results['true_labels'])
        analyze_compound_features(test_descriptors, results['true_labels'], descriptor_names)
        save_feature_mappings(descriptor_names)

        print("\n" + "="*60)
        print("Assessment completed! The result has been saved.")
        print("="*60)

    elif TASK == 'predict':
        print("\n" + "="*60)
        print("Prediction mode")
        print("="*60)

        print(f"Loading the data to be predicted: {PREDICTION_FILE}")
        try:
            smiles_list, protein_list = [], []
            with open(PREDICTION_FILE, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        smiles_list.append(parts[0])
                        protein_list.append(' '.join(parts[1:]))
            print(f"Successfully loaded {len(smiles_list)} samples")
        except Exception as e:
            print(f"Failed to load the prediction file: {str(e)}")
            return

        print("Recalculate the molecular descriptor...")
        
        metadata_path = 'descriptors/descriptor_metadata.json'
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(
                "Error: Descriptor metadata file not found"
            )
        
        ligand_features, valid_idx = compute_descriptors_for_prediction(smiles_list, metadata_path)
        valid_smiles = np.array(smiles_list)[valid_idx]
        valid_proteins = np.array(protein_list)[valid_idx]

        total_idx = np.arange(len(smiles_list))
        invalid_mask = np.ones(len(smiles_list), dtype=bool)
        invalid_mask[valid_idx] = False
        if invalid_mask.any():
            try:
                with open('invalid_samples.txt', 'w', encoding='utf-8') as f:
                    f.write("Invalid SMILES:\n")
                    for i in total_idx[invalid_mask]:
                        f.write(f"{smiles_list[i]}\t{protein_list[i]}\n")
                print("Save invalid SMILES to: invalid_samples.txt")
            except Exception as e:
                print(f"Failed to save invalid samples: {str(e)}")

        ligand_features = scaler.transform(ligand_features)

        encoded_proteins = np.array([protein_tokenizer.encode(p, config.protein_max_len) for p in valid_proteins])
        encoded_proteins = torch.LongTensor(encoded_proteins)
        ligand_features = torch.FloatTensor(ligand_features)

        predict_dataset = CPIDataset(encoded_proteins, ligand_features)
        predict_loader = DataLoader(predict_dataset, batch_size=config.batch_size, shuffle=False)

        print("Loading nn...")
        try:
            model = CPIPredictor(config, protein_tokenizer.vocab_size, ligand_features.shape[1]).to(device)
            model.load_state_dict(torch.load('models/final_nn_model.pth', map_location=device))
            model.eval()
        except Exception as e:
            print(f"Failed to load the model: {str(e)}")
            return

        probs = evaluate_model(model, predict_loader, device)

        print(f"Saving prediction result: {OUTPUT_FILE}")
        try:
            with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
                f.write("SMILES\tProtein\tPrediction\n")
                for s, p, pr in zip(valid_smiles, valid_proteins, probs):
                    f.write(f"{s}\t{p}\t{float(pr):.6f}\n")
        except Exception as e:
            print(f"Failed to save the prediction result: {str(e)}")
            return

        save_raw_data(valid_smiles, valid_proteins, None, probs)
        save_feature_mappings(descriptor_names)

        print("\n" + "="*60)
        print(f"Prediction complete! The result is saved to: {OUTPUT_FILE}")
        print(f"A total of {len(valid_smiles)} valid samples were predicted")
        print("="*60)

    else:
        print(f"Error: unknown task '{TASK}'. Use 'validate' or 'predict'。")

if __name__ == '__main__':
    main()