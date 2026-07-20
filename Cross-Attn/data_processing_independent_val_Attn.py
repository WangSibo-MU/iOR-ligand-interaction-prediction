import numpy as np
import os
import argparse

def process_data(file_path='dataset.txt', max_protein_len=480, max_smiles_len=70, random_state=42, balance_samples=False):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    smiles_list = []
    protein_list = []
    labels = []

    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 3:
            smiles = parts[0]
            protein = parts[1]
            label = int(parts[2])
            
            smiles_list.append(smiles)
            protein_list.append(protein)
            labels.append(label)

    actual_max_smiles = max(len(s) for s in smiles_list)
    actual_max_protein = max(len(p) for p in protein_list)
    
    print(f"Actual maximum SMILES length: {actual_max_smiles}")
    print(f"Actual maximum protein length: {actual_max_protein}")
    print(f"Using SMILES length: {max_smiles_len}")
    print(f"Using protein length: {max_protein_len}")

    processed_smiles = []
    for smile in smiles_list:
        if len(smile) > max_smiles_len:
            processed = smile[:max_smiles_len]
        else:
            processed = smile.ljust(max_smiles_len)
        processed_smiles.append(processed)

    processed_proteins = []
    for protein in protein_list:
        if len(protein) > max_protein_len:
            processed = protein[:max_protein_len]
        else:
            processed = protein.ljust(max_protein_len)
        processed_proteins.append(processed)

    smiles_array = np.array(processed_smiles)
    protein_array = np.array(processed_proteins)
    labels_array = np.array(labels)

    # ================= 新增：外部验证集内动态强制平衡 =================
    if balance_samples:
        print("\n[Balance] Applying dynamic balancing to External Validation set...")
        rng = np.random.default_rng(random_state)

        pos_idx = np.where(labels_array == 1)[0]
        neg_idx = np.where(labels_array == 0)[0]
        min_count = min(len(pos_idx), len(neg_idx))
        
        pos_sampled = rng.choice(pos_idx, min_count, replace=False)
        neg_sampled = rng.choice(neg_idx, min_count, replace=False)
        balanced_idx = np.concatenate([pos_sampled, neg_sampled])
        rng.shuffle(balanced_idx)
        
        smiles_array = smiles_array[balanced_idx]
        protein_array = protein_array[balanced_idx]
        labels_array = labels_array[balanced_idx]
    # ==============================================================

    os.makedirs('processed_data', exist_ok=True)

    np.savez('processed_data/external_validation_Slit.npz', 
             smiles=smiles_array, 
             proteins=protein_array, 
             labels=labels_array)

    with open('processed_data/external_validation.txt', 'w') as f:
        for s, p, l in zip(smiles_array, protein_array, labels_array):
            f.write(f"{s.strip()} {p.strip()} {l}\n")
    
    print(f"\nData processing completed, saved to processed_data directory:")
    print(f"- Dataset Type: Independent External Validation Set (独立外部验证集)")
    print(f"- Balanced Mode: {'Enabled (Strict 1:1)' if balance_samples else 'Disabled'}")
    print(f"- External validation set: {len(smiles_array)} samples (Pos: {np.sum(labels_array==1)}, Neg: {np.sum(labels_array==0)})")
    
    return {
        'external_validation': {
            'smiles': smiles_array,
            'proteins': protein_array,
            'labels': labels_array
        }
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process compound-protein interaction data for independent external validation.")
    parser.add_argument('--file_path', type=str, default='dataset.txt', help='Path to the dataset text file.')
    parser.add_argument('--balance_samples', action='store_true', help='Force equal number of positive and negative samples via downsampling.')
    parser.add_argument('--max_protein_len', type=int, default=480, help='Maximum truncated/padded length for proteins.')
    parser.add_argument('--max_smiles_len', type=int, default=70, help='Maximum truncated/padded length for SMILES.')
    parser.add_argument('--random_state', type=int, default=42, help='Random state seed.')
    
    args = parser.parse_args()

    process_data(
        file_path=args.file_path,
        max_protein_len=args.max_protein_len,
        max_smiles_len=args.max_smiles_len,
        random_state=args.random_state,
        balance_samples=args.balance_samples
    )