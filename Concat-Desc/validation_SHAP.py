import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, recall_score, matthews_corrcoef
import joblib
import warnings
import os
from tqdm import tqdm
from utils import Config, CharTokenizer, CPIDataset, CPIPredictor
from rdkit.Chem import Descriptors
import shap
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')
os.environ['LOKY_MAX_CPU_COUNT'] = '1000'

# =======================================================
USE_GPU = True  
SHAP_CONFIG = {
    'do_global_shap': False,           
    'do_local_shap': True,            
    'sample_indices': [608, 609, 610],  #Fill in the position of the target visualization sample in the dataset here
    'compute_all_shap': False,        
    'background_size': 50             
}
# =====================================================

RESULTS_DIR = "results"
SHAP_DIR = "SHAP"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(SHAP_DIR, exist_ok=True)

class CompoundModelWrapper:
    def __init__(self, model, device, protein_tensor):
        self.model = model
        self.device = device
        self.protein_tensor = protein_tensor.to(device)
        
    def __call__(self, ligand_features):
        ligand_tensor = torch.FloatTensor(ligand_features).to(self.device)
        batch_size = ligand_tensor.shape[0]
        protein_batch = self.protein_tensor.repeat(batch_size, 1)
        with torch.no_grad():
            probs = self.model(protein_batch, ligand_tensor).view(-1)   
            probs = torch.clamp(probs, 1e-6, 1 - 1e-6)                
            logits = torch.logit(probs)  
            return logits.cpu().numpy()

def evaluate_model(model, data_loader, device, model_name="Final Model"):
    model.eval()
    predictions = []
    true_labels = []
    probabilities = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            protein, ligand_features, labels = batch
            protein = protein.to(device)
            ligand_features = ligand_features.float().to(device)
            true_labels.extend(labels.cpu().numpy())
            
            outputs = model(protein, ligand_features)
            probs = outputs.cpu().numpy().astype(float)
            if probs.ndim == 0:
                probs = [probs]
            probabilities.extend(probs)
            
            batch_preds = (np.array(probs) > 0.5).astype(int)
            predictions.extend(batch_preds)

    probabilities = np.array(probabilities)
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)

    auc = roc_auc_score(true_labels, probabilities) if len(np.unique(true_labels)) > 1 else 0.5
    acc = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions) if len(np.unique(true_labels)) > 1 else 0
    recall = recall_score(true_labels, predictions) if np.sum(true_labels) > 0 else 0
    mcc = matthews_corrcoef(true_labels, predictions) if len(np.unique(true_labels)) > 1 else 0

    results_path = os.path.join(RESULTS_DIR, 'evaluation_results.txt')
    with open(results_path, 'w') as f:
        f.write(f"Model Name: {model_name}\n")
        f.write(f"Test Set Size: {len(true_labels)} samples\n")
        f.write("====================\n")
        f.write(f"AUC: {auc:.4f}\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"Matthews Correlation Coefficient (MCC): {mcc:.4f}\n")
        f.write("====================\n")

    print(f"Evaluation results saved to {results_path}")

    return {
        'auc': auc,
        'accuracy': acc,
        'f1': f1,
        'recall': recall,
        'mcc': mcc
    }

def perform_shap_analysis(model, test_loader, device, descriptor_names, shap_config):
    print("Preparing for SHAP analysis...")
    all_ligand_features = []
    all_labels = []
    all_proteins = []

    for protein, ligand_features, labels in test_loader:
        all_ligand_features.append(ligand_features.numpy())
        all_labels.append(labels.numpy())
        all_proteins.append(protein.numpy())

    ligand_features_np = np.vstack(all_ligand_features)
    labels_np = np.concatenate(all_labels)
    proteins_np = np.vstack(all_proteins) 

    background_size = min(shap_config['background_size'], ligand_features_np.shape[0])
    np.random.seed(42)
    random_indices = np.random.choice(ligand_features_np.shape[0], size=background_size, replace=False)
    background_data = ligand_features_np[random_indices]
    fixed_protein_tensor = torch.from_numpy(proteins_np[:1])
    global_wrapper = CompoundModelWrapper(model, device, fixed_protein_tensor)

    global_explainer = shap.Explainer(
        global_wrapper,
        background_data,
        feature_names=descriptor_names
    )

    if shap_config['compute_all_shap']:
        shap_values_global = global_explainer(ligand_features_np)
        sample_data_for_global = ligand_features_np
        shap_df = pd.DataFrame(ligand_features_np, columns=descriptor_names)
        shap_df['label'] = labels_np
        shap_df.to_csv(os.path.join(SHAP_DIR, 'all_samples_data.csv'), index=False)
        shap_values_df = pd.DataFrame(shap_values_global.values, columns=descriptor_names)
        shap_values_df.to_csv(os.path.join(SHAP_DIR, 'all_samples_shap_values.csv'), index=False)
        print(f"Saved SHAP values for all {len(ligand_features_np)} samples")
    else:
        valid_indices_for_global = [i for i in shap_config['sample_indices'] if i < ligand_features_np.shape[0]]
        sample_data_for_global = ligand_features_np[valid_indices_for_global]
        shap_values_global = global_explainer(sample_data_for_global)
        print(f"Computed SHAP values for samples (global approx): {valid_indices_for_global}")

    if shap_config['do_global_shap']:
        print("Performing global swap analysis...")
        global_shap_values = np.abs(shap_values_global.values).mean(0)
        feature_importance = pd.DataFrame({
            'feature': descriptor_names,
            'importance': global_shap_values
        }).sort_values('importance', ascending=False)
        feature_importance.to_csv(os.path.join(SHAP_DIR, 'global_feature_importance.csv'), index=False)

        plt.figure(figsize=(10, 8))
        sns.barplot(x='importance', y='feature', data=feature_importance.head(20))
        plt.title('Global Feature Importance (SHAP)')
        plt.tight_layout()
        plt.savefig(os.path.join(SHAP_DIR, 'global_feature_importance.png'), dpi=300)
        plt.close()

        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values_global.values, sample_data_for_global,
                          feature_names=descriptor_names, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(SHAP_DIR, 'shap_summary_plot.png'), dpi=300)
        plt.close()
        print("The global SHAP analysis is completed and the results are saved")

    if shap_config['do_local_shap'] and shap_config['sample_indices']:
        print("Perform local shap analysis...")
        valid_indices = [i for i in shap_config['sample_indices'] if i < ligand_features_np.shape[0]]
        if not valid_indices:
            print("No valid sample indices provided for local SHAP analysis")
            return

        for idx in valid_indices:
            protein_tensor_i = torch.from_numpy(proteins_np[idx:idx+1]) 
            wrapper_i = CompoundModelWrapper(model, device, protein_tensor_i)
            explainer_i = shap.Explainer(wrapper_i, background_data, feature_names=descriptor_names)
            sample_feat = ligand_features_np[idx:idx+1]
            shap_values_i = explainer_i(sample_feat)
            ev = shap_values_i[0]
            sample_explanation = shap.Explanation(
                values=ev.values,
                base_values=ev.base_values, 
                data=sample_feat[0],
                feature_names=descriptor_names
            )

            sample_df = pd.DataFrame({
                'feature': descriptor_names,
                'value': sample_feat[0],
                'shap_value': ev.values
            })
            sample_df.to_csv(os.path.join(SHAP_DIR, f'sample_{idx}_data.csv'), index=False)

            plt.figure(figsize=(12, 8))
            shap.plots.waterfall(sample_explanation, max_display=15, show=False)
            plt.title(f'SHAP Waterfall Plot for Sample {idx}', fontsize=16)
            plt.tight_layout()
            plt.savefig(os.path.join(SHAP_DIR, f'shap_waterfall_sample_{idx}.png'),
                        dpi=300, bbox_inches='tight')
            plt.close()

            print(f"Local SHAP analysis completed {idx}")

    print("SHAP analysis completed")

def main():
    if USE_GPU and torch.cuda.is_available():
        device = torch.device('cuda')
        print("Use GPU")
    else:
        device = torch.device('cpu')
        print("Use CPU")
    
    config = Config()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(SHAP_DIR, exist_ok=True)
    
    print("\n" + "="*60)
    print("Run validation mode")
    print("="*60)

    try:
        with open('models/protein_tokenizer_chars.txt', 'r') as f:
            protein_chars = f.read().strip().split(',')
        protein_tokenizer = CharTokenizer(protein_chars)
        print(f"Loading protein tokenizer, including {len(protein_chars)} tokens")
    except FileNotFoundError:
        print("Error: can't find 'models/protein_tokenizer_chars.txt'")
        return

    try:
        scaler = joblib.load('models/ligand_scaler.pkl')
        print("Loading molecular descriptor normalizer")
    except FileNotFoundError:
        print("Error: can't find 'models/ligand_scaler.pkl'")
        return

    vif_names_path = 'descriptors/selected_descriptor_names_vif.txt'
    if os.path.exists(vif_names_path):
        with open(vif_names_path, 'r') as f:
            descriptor_names = f.read().strip().split(',')
        print(f"Loading {len(descriptor_names)} de redundant descriptor")
    else:
        raise FileNotFoundError(
            "Error: list of de redundant descriptors not found"
            "Please run the data preprocessing script first"
        )

    print("loading test data...")
    try:
        test_data = np.load('processed_data/test.npz', allow_pickle=True)
        test_smiles = test_data['smiles']
        test_proteins = test_data['proteins']
        test_labels = test_data['labels']
        test_descriptors = test_data['descriptors']
    except Exception as e:
        print(f"Failed to load test data from 'processed_data/test.npz': {str(e)}")
        return

    print(f"Test set: {len(test_smiles)} samples")

    test_descriptors = scaler.transform(test_descriptors)

    encoded_proteins = np.array([protein_tokenizer.encode(p, config.protein_max_len) for p in test_proteins])
    encoded_proteins_tensor = torch.LongTensor(encoded_proteins)
    ligand_features_tensor = torch.FloatTensor(test_descriptors)
    test_labels_tensor = torch.FloatTensor(test_labels)

    test_dataset = CPIDataset(encoded_proteins_tensor, ligand_features_tensor, test_labels_tensor)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    print("Loading pretrained model...")
    try:
        model = CPIPredictor(
            config, 
            protein_tokenizer.vocab_size, 
            ligand_features_tensor.shape[1]
        ).to(device)
        model.load_state_dict(torch.load('models/final_nn_model.pth', map_location=device))
        model.eval()
    except Exception as e:
        print(f"Loading model 'models/final_nn_model.pth' failed: {str(e)}")
        return

    print("\nStart model evaluation...")
    results = evaluate_model(model, test_loader, device)
    
    print("\n" + "="*60)
    print("Evaluation summary")
    for metric, value in results.items():
        print(f"{metric.upper()}: {value:.4f}")
    print("="*60)
    
    print(f"\nValidation complete. Results saved at '{os.path.join(RESULTS_DIR, 'evaluation_results.txt')}'")

    if SHAP_CONFIG['do_global_shap'] or SHAP_CONFIG['do_local_shap']:
        print("\n" + "="*60)
        print("Start SHAP analysis")
        print("="*60)
        perform_shap_analysis(model, test_loader, device, descriptor_names, SHAP_CONFIG)

if __name__ == '__main__':
    main()