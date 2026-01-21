import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import os
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.feature_selection import VarianceThreshold
import json
import time

def compute_molecular_descriptors(smiles_list, descriptor_names=None):
    if descriptor_names is None:
        descriptor_names = [desc[0] for desc in Descriptors._descList]
    
    calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)
    
    descriptors = []
    valid_indices = []
    failed_count = 0
    
    print(f"Starting calculation of descriptors for {len(smiles_list)} molecules...")
    for idx, smiles in enumerate(smiles_list):
        if idx % 500 == 0 and idx > 0:
            print(f"  processed {idx}/{len(smiles_list)} molecules...")
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            try:
                desc_values = calculator.CalcDescriptors(mol)
                descriptors.append(desc_values)
                valid_indices.append(idx)
            except Exception as e:
                failed_count += 1
                if failed_count <= 5:
                    print(f"  [Error] Calculation failed: {smiles[:50]}... Error: {str(e)[:50]}")
                continue
        else:
            failed_count += 1
            if failed_count <= 5:
                print(f"  [Invalid] SMILES: {smiles[:50]}...")

    descriptors = np.array(descriptors)
    print(f"  ✓ Successed: {len(valid_indices)}, ✗ failed: {failed_count}")
    return descriptors, valid_indices

def filter_descriptors_by_variance_and_vif(descriptor_csv_path, 
                                          variance_threshold=0.0, 
                                          vif_threshold=5.0,
                                          max_iterations=50,
                                          min_features=50):
    print("\n" + "="*70)
    print(f"Start variance and VIF filtering...")
    print(f"Input: {descriptor_csv_path}")
    print("="*70)
    
    start_time = time.time()
    
    print(f"\n[1/3] Loading descriptors...")
    df = pd.read_csv(descriptor_csv_path)
    
    smiles_col = df['SMILES']
    descriptor_matrix = df.drop(columns=['SMILES']).values
    
    original_n_samples, original_n_features = descriptor_matrix.shape
    print(f"  Raw: {original_n_samples} samples, {original_n_features} descriptors")
    
    print(f"\n[2/3] Applying variance filter (threshold={variance_threshold})...")
    variance_selector = VarianceThreshold(threshold=variance_threshold)
    descriptor_matrix = variance_selector.fit_transform(descriptor_matrix)
    selected_indices = variance_selector.get_support(indices=True).tolist()
    remaining_names = df.drop(columns=['SMILES']).columns[selected_indices].tolist()
    
    n_after_variance = descriptor_matrix.shape[1]
    print(f"  After variance filter: {n_after_variance} descriptors retained")
    print(f"  Removed {original_n_features - n_after_variance} descriptors")
    
    print(f"\n[3/3] Using VIF filtering (threshold={vif_threshold}, minimum retained={min_features})...")
    iteration = 0
    vif_exceeded = True
    removed_features = []
    
    while (vif_exceeded and iteration < max_iterations and 
           descriptor_matrix.shape[1] > min_features):
        iteration += 1
        print(f"\n  --- VIF iteration {iteration} ---")
        print(f"  Current feature count: {descriptor_matrix.shape[1]}")

        vif_values = []
        for i in range(descriptor_matrix.shape[1]):
            try:
                vif = variance_inflation_factor(descriptor_matrix, i)
                vif_values.append(vif)
            except Exception as e:
                print(f"  Warning: Feature {i} calculation failed: {e}")
                vif_values.append(np.inf)

        max_vif_idx = np.argmax(vif_values)
        max_vif = vif_values[max_vif_idx]
        
        print(f"  Max VIF: {max_vif:.2f} (feature '{remaining_names[max_vif_idx]}')")
        
        if max_vif > vif_threshold and descriptor_matrix.shape[1] > min_features + 1:
            print(f"  → Removing features '{remaining_names[max_vif_idx]}'")
            removed_features.append({
                'name': remaining_names[max_vif_idx],
                'vif': max_vif,
                'iteration': iteration
            })
            descriptor_matrix = np.delete(descriptor_matrix, max_vif_idx, axis=1)
            del remaining_names[max_vif_idx]
        else:
            vif_exceeded = False
            print(f"  ✓ The VIF of all remaining features <= {vif_threshold}")
        
        if descriptor_matrix.shape[1] <= min_features:
            print(f"  ⚠ Reach the minimum feature number limit ({min_features})")
            break
    
    elapsed_time = time.time() - start_time
    
    final_n_features = descriptor_matrix.shape[1]
    print("\n" + "="*70)
    print("VIF filtering completed!")
    print(f"Filtering time: {elapsed_time:.2f} seconds")
    print(f"Ultimately retained: {final_n_features} descriptors (Retention rate: {100*final_n_features/original_n_features:.1f}%)")
    print(f"Total removed: {original_n_features - final_n_features} descriptors")
    
    if removed_features:
        print(f"\nVIF filter removal details (Top 10):")
        for feat in removed_features[:10]:
            print(f"  - {feat['name']:<40} VIF={feat['vif']:.2f} (iteration {feat['iteration']})")
        if len(removed_features) > 10:
            print(f"  ... left {len(removed_features) - 10}")
    print("="*70)

    filtered_df = pd.DataFrame(descriptor_matrix, columns=remaining_names)
    filtered_df.insert(0, 'SMILES', smiles_col.values[:len(filtered_df)])

    filtered_csv_path = descriptor_csv_path.replace('.csv', '_filtered_vif.csv')
    filtered_df.to_csv(filtered_csv_path, index=False)

    all_feature_names = df.drop(columns=['SMILES']).columns.tolist()
    selected_indices_final = [all_feature_names.index(name) for name in remaining_names]
    
    np.save('descriptors/selected_descriptor_indices_vif.npy', selected_indices_final)
    with open('descriptors/selected_descriptor_names_vif.txt', 'w') as f:
        f.write(','.join(remaining_names))
    
    print(f"\nFile saved:")
    print(f"  - Filtered descriptors: {filtered_csv_path}")
    print(f"  - Feature indices: descriptors/selected_descriptor_indices_vif.npy")
    print(f"  - Feature names: descriptors/selected_descriptor_names_vif.txt")
    
    return descriptor_matrix, selected_indices_final, remaining_names

def process_data(file_path, max_protein_len=480, test_size=0.2, random_state=42,
                 variance_threshold=0.001, vif_threshold=10.0, max_iterations=150, min_features=50):

    print("\n" + "="*70)
    print("Data preprocessing pipeline started")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    start_time = time.time()

    print("\n[1/7] Loading raw dataset...")
    with open(file_path, 'r') as f:
        lines = f.readlines()
    print(f"  Loaded {len(lines)} lines of data")

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

    print(f"  Valid data: {len(smiles_list)} samples")
    print("\n[2/7] Analyzing protein sequence features...")
    all_protein_chars = set()
    for p in protein_list:
        all_protein_chars.update(p)
    protein_chars = sorted(all_protein_chars)
    
    print(f"  Amino acid character set size: {len(protein_chars)}")
    print(f"  Character: {', '.join(protein_chars)}")

    actual_max_smiles = max(len(s) for s in smiles_list)
    actual_max_protein = max(len(p) for p in protein_list)
    
    print(f"\n  SMILES length - actual maximum: {actual_max_smiles}")
    print(f"  Protein length - actual maximum: {actual_max_protein}")
    print(f"  Protein length - setting used: {max_protein_len}")
    print("\n[3/7] Calculating molecular descriptors...")
    
    all_descriptor_names = [desc[0] for desc in Descriptors._descList]
    print(f"  Total number of descriptors available: {len(all_descriptor_names)}")
    
    os.makedirs('descriptors', exist_ok=True)
    with open('descriptors/all_descriptor_names.json', 'w') as f:
        json.dump(all_descriptor_names, f)
    
    full_descriptors, valid_indices = compute_molecular_descriptors(smiles_list, all_descriptor_names)
    print(f"  Successfully calculated descriptors for {len(valid_indices)} samples")
    
    smiles_list = [smiles_list[i] for i in valid_indices]
    protein_list = [protein_list[i] for i in valid_indices]
    labels = np.array(labels)[valid_indices].tolist()
    
    print(f"  {len(smiles_list)} valid samples remain after filtering")
    print("\n[4/7] Splitting train/test sets...")
    
    indices = np.arange(len(smiles_list))
    
    train_idx, test_idx = train_test_split(
        indices, 
        test_size=test_size, 
        random_state=random_state,
    )
    
    smiles_train = np.array(smiles_list, dtype=object)[train_idx]
    protein_train = np.array(protein_list)[train_idx]
    labels_train = np.array(labels)[train_idx]
    
    smiles_test = np.array(smiles_list, dtype=object)[test_idx]
    protein_test = np.array(protein_list)[test_idx]
    labels_test = np.array(labels)[test_idx]

    print(f"  Train set: {len(smiles_train)} samples ({100*(1-test_size):.0f}%)")
    print(f"  Test set: {len(smiles_test)} samples ({100*test_size:.0f}%)")
    print(f"  ✓ Split completed, no duplicate sample issues")

    print("\n[5/7] Applying Variance + VIF filtering on training set...")
    
    desc_df = pd.DataFrame(full_descriptors, columns=all_descriptor_names)
    desc_df.insert(0, 'SMILES', smiles_list)
    desc_df.to_csv('descriptors/all_descriptors_raw.csv', index=False)
    print(f"  Saving all descriptors to: descriptors/all_descriptors_raw.csv")
    
    train_descriptors = full_descriptors[train_idx]
    
    print(f"  Training set descriptor shape: {train_descriptors.shape}")
    print(f"  Training set SMILES count: {len(smiles_train)}")
    assert train_descriptors.shape[0] == len(smiles_train), f"Error: Count mismatch! {train_descriptors.shape[0]} vs {len(smiles_train)}"
    print("  ✓ Data consistency check passed")
    
    train_desc_df = pd.DataFrame(train_descriptors, columns=all_descriptor_names)
    train_desc_df.insert(0, 'SMILES', smiles_train)
    train_desc_df.to_csv('descriptors/train_descriptors_raw.csv', index=False)
    print(f"  Saving training set descriptors to: descriptors/train_descriptors_raw.csv")
    
    filtered_descriptors, feature_indices, feature_names = filter_descriptors_by_variance_and_vif(
        descriptor_csv_path='descriptors/train_descriptors_raw.csv',
        variance_threshold=variance_threshold,
        vif_threshold=vif_threshold,
        max_iterations=max_iterations,
        min_features=min_features
    )
    
    metadata = {
        'all_descriptor_names': all_descriptor_names,
        'selected_indices': feature_indices,
        'selected_names': feature_names,
        'variance_threshold': variance_threshold,
        'vif_threshold': vif_threshold
    }
    with open('descriptors/descriptor_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  Metadata saved: descriptors/descriptor_metadata.json")
    
    all_descriptors_filtered = full_descriptors[:, feature_indices]
    
    print(f"  {len(feature_names)} descriptors retained after filtering")
    print("\n[6/7] Saving filtered descriptors...")
    
    desc_df_filtered = pd.DataFrame(all_descriptors_filtered, columns=feature_names)
    desc_df_filtered.insert(0, 'SMILES', smiles_list)
    desc_df_filtered.to_csv('descriptors/molecular_descriptors_filtered.csv', index=False)
    print(f"  File: descriptors/molecular_descriptors_filtered.csv")

    print("\n[7/7] Processing protein sequences and saving final data...")
    
    os.makedirs('models', exist_ok=True)
    with open('models/protein_tokenizer_chars.txt', 'w') as f:
        f.write(','.join(protein_chars))

    processed_proteins = []
    for i, protein in enumerate(protein_list):
        if len(protein) > max_protein_len:
            processed = protein[:max_protein_len]
        else:
            processed = protein.ljust(max_protein_len)
        processed_proteins.append(processed)
    
    protein_array_processed = np.array(processed_proteins)

    os.makedirs('processed_data', exist_ok=True)

    train_desc_filtered = all_descriptors_filtered[train_idx]
    test_desc_filtered = all_descriptors_filtered[test_idx]
    
    print(f"  Saving training set data...")
    np.savez('processed_data/train.npz', 
             smiles=smiles_train, 
             proteins=protein_train, 
             labels=labels_train,
             descriptors=train_desc_filtered,
             allow_pickle=True)
    print(f"    → processed_data/train.npz")
    
    print(f"  Saving test set data...")
    np.savez('processed_data/test.npz', 
             smiles=smiles_test, 
             proteins=protein_test, 
             labels=labels_test,
             descriptors=test_desc_filtered,
             allow_pickle=True)
    print(f"    → processed_data/test.npz")

    with open('processed_data/train.txt', 'w') as f_train:
        for s, p, l in zip(smiles_train, protein_train, labels_train):
            f_train.write(f"{s} {p.strip()} {l}\n")
    
    with open('processed_data/test.txt', 'w') as f_test:
        for s, p, l in zip(smiles_test, protein_test, labels_test):
            f_test.write(f"{s} {p.strip()} {l}\n")
    print(f"  txt has been saved")
    
    total_time = time.time() - start_time
    
    print("\n" + "="*70)
    print("Data preprocessing completed!")
    print(f"Total time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
    print("="*70)
    print(f"Training set: {len(smiles_train)} samples")
    print(f"Test set: {len(smiles_test)} samples")
    print(f"\nOutput files:")
    print(f"  - Training set: processed_data/train.npz, train.txt")
    print(f"  - Test set: processed_data/test.npz, test.txt")
    print(f"  - Descriptors: descriptors/molecular_descriptors_filtered.csv")
    print(f"  - Feature indices: descriptors/selected_descriptor_indices_vif.npy")
    print(f"  - metadata: descriptors/descriptor_metadata.json")
    
    return {
        'train': {'smiles': smiles_train, 'proteins': protein_train, 'labels': labels_train},
        'test': {'smiles': smiles_test, 'proteins': protein_test, 'labels': labels_test},
        'actual_max_smiles': actual_max_smiles,
        'actual_max_protein': actual_max_protein,
        'filtered_descriptor_names': feature_names
    }

if __name__ == "__main__":
    # Using custom parameters
    file_path = 'dataset.txt'
    
    print("\n" + "="*70)
    print("Configuration parameters")
    print("="*70)
    print(f"Dataset file: {file_path}")
    print(f"Test set ratio: {0.2} (train set: {0.8})")
    print(f"Random seed: {42}")
    print(f"Variance threshold: {0.001}")
    print(f"VIF threshold: {10.0}")
    print(f"Max epochs: {150}")
    print(f"Min number of features: {50}")
    
    processed_data = process_data(
        file_path,
        variance_threshold=0.001,
        vif_threshold=10.0,
        max_iterations=150,
        min_features=50
    )