import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import os
import argparse
from collections import defaultdict
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

def _coerce_descriptor_numeric_frame(X_df):
    """
    Convert a descriptor dataframe to numeric float values and normalize non-finite
    values. RDKit descriptors can occasionally produce NaN/inf for special salts,
    ions, or unusual structures. VIF must never receive NaN/inf.
    """
    X_df = X_df.apply(pd.to_numeric, errors='coerce')
    X_df = X_df.replace([np.inf, -np.inf], np.nan)
    return X_df


def _summarize_missing_descriptor_values(X_df):
    """Return columns containing NaN after coercion/replacement and their counts."""
    missing_counts = X_df.isna().sum()
    missing_counts = missing_counts[missing_counts > 0].sort_values(ascending=False)
    return {str(k): int(v) for k, v in missing_counts.items()}


def apply_descriptor_preprocessing(descriptor_array, all_descriptor_names, selected_names, preprocess_info):
    """
    Apply the descriptor cleaning learned from the training set to any descriptor
    matrix, then return only the selected descriptor columns.

    This function is used for both train and test sets so that the test set is
    processed with exactly the same feature list and median-imputation values as
    the training set.
    """
    X_df = pd.DataFrame(descriptor_array, columns=all_descriptor_names)
    X_df = _coerce_descriptor_numeric_frame(X_df)

    missing_selected = [name for name in selected_names if name not in X_df.columns]
    if missing_selected:
        raise ValueError(f"Selected descriptors are missing from input data: {missing_selected[:20]}")

    X_selected = X_df.loc[:, selected_names].copy()
    impute_values = preprocess_info.get('impute_values', {})

    for col in selected_names:
        fill_value = float(impute_values.get(col, 0.0))
        X_selected[col] = X_selected[col].fillna(fill_value)

    matrix = X_selected.to_numpy(dtype=np.float64)
    if not np.isfinite(matrix).all():
        bad_rows, bad_cols = np.where(~np.isfinite(matrix))
        bad_feature_names = sorted(set(selected_names[j] for j in bad_cols))
        raise ValueError(
            "Descriptor matrix still contains NaN/inf after preprocessing. "
            f"Bad columns: {bad_feature_names[:20]}"
        )

    return matrix


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
    
    print(f"\n[1/4] Loading and cleaning descriptors...")
    df = pd.read_csv(descriptor_csv_path)

    if 'SMILES' not in df.columns:
        raise ValueError("Descriptor CSV must contain a 'SMILES' column.")
    
    smiles_col = df['SMILES']
    original_feature_names = df.drop(columns=['SMILES']).columns.tolist()
    X_df = df.drop(columns=['SMILES']).copy()
    original_n_samples, original_n_features = X_df.shape
    print(f"  Raw: {original_n_samples} samples, {original_n_features} descriptors")

    X_df = _coerce_descriptor_numeric_frame(X_df)
    missing_counts_before = _summarize_missing_descriptor_values(X_df)

    if missing_counts_before:
        print(f"  Columns containing NaN/inf after RDKit descriptor calculation: {len(missing_counts_before)}")
        print(f"  First problematic columns: {list(missing_counts_before.keys())[:20]}")
    else:
        print("  No NaN/inf values detected before VIF filtering.")

    # Drop descriptors that are completely unavailable in the training set.
    all_nan_cols = [col for col in X_df.columns if X_df[col].isna().all()]
    if all_nan_cols:
        print(f"  Dropping all-NaN descriptor columns: {all_nan_cols[:20]}")
        X_df = X_df.drop(columns=all_nan_cols)

    if X_df.shape[1] == 0:
        raise ValueError("No descriptor columns remain after dropping all-NaN columns.")

    # Median imputation is fitted only on the training descriptors.
    medians = X_df.median(axis=0, skipna=True)
    medians = medians.fillna(0.0)
    X_df = X_df.fillna(medians)

    impute_values = {str(col): float(medians[col]) for col in X_df.columns}
    cleaned_feature_names = X_df.columns.tolist()
    descriptor_matrix = X_df.to_numpy(dtype=np.float64)

    if not np.isfinite(descriptor_matrix).all():
        bad_rows, bad_cols = np.where(~np.isfinite(descriptor_matrix))
        bad_feature_names = sorted(set(cleaned_feature_names[j] for j in bad_cols))
        raise ValueError(
            "Descriptor matrix contains NaN/inf after cleaning. "
            f"Bad columns: {bad_feature_names[:20]}"
        )

    print(f"  After cleaning: {descriptor_matrix.shape[1]} descriptors retained")
    print(f"  Dropped all-NaN descriptors: {len(all_nan_cols)}")
    print(f"  Missing values filled by training-set median: {sum(missing_counts_before.values())}")
    
    print(f"\n[2/4] Applying variance filter (threshold={variance_threshold})...")
    variance_selector = VarianceThreshold(threshold=variance_threshold)
    descriptor_matrix = variance_selector.fit_transform(descriptor_matrix)
    selected_indices_after_cleaning = variance_selector.get_support(indices=True).tolist()
    remaining_names = [cleaned_feature_names[i] for i in selected_indices_after_cleaning]
    
    n_after_variance = descriptor_matrix.shape[1]
    print(f"  After variance filter: {n_after_variance} descriptors retained")
    print(f"  Removed {len(cleaned_feature_names) - n_after_variance} descriptors by variance filter")

    if descriptor_matrix.shape[1] == 0:
        raise ValueError("No descriptor columns remain after variance filtering.")
    
    print(f"\n[3/4] Using VIF filtering (threshold={vif_threshold}, minimum retained={min_features})...")
    iteration = 0
    removed_features = []
    
    while iteration < max_iterations and descriptor_matrix.shape[1] > min_features:
        iteration += 1
        print(f"\n  --- VIF iteration {iteration} ---")
        print(f"  Current feature count: {descriptor_matrix.shape[1]}")

        if not np.isfinite(descriptor_matrix).all():
            bad_rows, bad_cols = np.where(~np.isfinite(descriptor_matrix))
            bad_feature_names = sorted(set(remaining_names[j] for j in bad_cols))
            raise ValueError(
                "VIF input contains NaN/inf. "
                f"Bad columns: {bad_feature_names[:20]}"
            )

        vif_values = []
        for i in range(descriptor_matrix.shape[1]):
            try:
                with np.errstate(divide='ignore', invalid='ignore'):
                    vif = variance_inflation_factor(descriptor_matrix, i)

                # Perfect multicollinearity can legitimately produce inf.
                # NaN is not acceptable as a successful VIF value, so treat it
                # as infinite and remove the corresponding feature.
                if not np.isfinite(vif):
                    vif = np.inf

                vif_values.append(float(vif))
            except Exception as e:
                print(f"  Warning: Feature '{remaining_names[i]}' VIF calculation failed: {e}")
                vif_values.append(np.inf)

        vif_values = np.asarray(vif_values, dtype=np.float64)
        max_vif_idx = int(np.argmax(vif_values))
        max_vif = float(vif_values[max_vif_idx])
        max_vif_display = "inf" if np.isinf(max_vif) else f"{max_vif:.2f}"
        
        print(f"  Max VIF: {max_vif_display} (feature '{remaining_names[max_vif_idx]}')")
        
        if max_vif > vif_threshold and descriptor_matrix.shape[1] > min_features:
            print(f"  → Removing feature '{remaining_names[max_vif_idx]}'")
            removed_features.append({
                'name': remaining_names[max_vif_idx],
                'vif': None if np.isinf(max_vif) else max_vif,
                'vif_is_infinite': bool(np.isinf(max_vif)),
                'iteration': iteration
            })
            descriptor_matrix = np.delete(descriptor_matrix, max_vif_idx, axis=1)
            del remaining_names[max_vif_idx]
        else:
            print(f"  ✓ All remaining finite VIF values are <= {vif_threshold}")
            break
        
        if descriptor_matrix.shape[1] <= min_features:
            print(f"  ⚠ Reached the minimum feature number limit ({min_features})")
            break

    if iteration >= max_iterations:
        print(f"  ⚠ Reached max_iterations={max_iterations}; remaining features may still have high VIF.")
    
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
            vif_text = "inf" if feat.get('vif_is_infinite') else f"{feat['vif']:.2f}"
            print(f"  - {feat['name']:<40} VIF={vif_text} (iteration {feat['iteration']})")
        if len(removed_features) > 10:
            print(f"  ... left {len(removed_features) - 10}")
    print("="*70)

    filtered_df = pd.DataFrame(descriptor_matrix, columns=remaining_names)
    filtered_df.insert(0, 'SMILES', smiles_col.values[:len(filtered_df)])

    filtered_csv_path = descriptor_csv_path.replace('.csv', '_filtered_vif.csv')
    filtered_df.to_csv(filtered_csv_path, index=False)

    selected_indices_final = [original_feature_names.index(name) for name in remaining_names]

    preprocess_info = {
        'original_feature_names': original_feature_names,
        'cleaned_feature_names': cleaned_feature_names,
        'dropped_all_nan_columns': all_nan_cols,
        'missing_counts_before_imputation': missing_counts_before,
        'impute_values': impute_values,
        'selected_names': remaining_names,
        'selected_indices': selected_indices_final,
        'removed_by_vif': removed_features,
    }
    
    os.makedirs('descriptors', exist_ok=True)
    np.save('descriptors/selected_descriptor_indices_vif.npy', selected_indices_final)
    with open('descriptors/selected_descriptor_names_vif.txt', 'w') as f:
        f.write(','.join(remaining_names))
    with open('descriptors/descriptor_preprocessing_info.json', 'w') as f:
        json.dump(preprocess_info, f, indent=2)
    
    print(f"\nFile saved:")
    print(f"  - Filtered descriptors: {filtered_csv_path}")
    print(f"  - Feature indices: descriptors/selected_descriptor_indices_vif.npy")
    print(f"  - Feature names: descriptors/selected_descriptor_names_vif.txt")
    print(f"  - Descriptor preprocessing info: descriptors/descriptor_preprocessing_info.json")
    
    return descriptor_matrix, selected_indices_final, remaining_names, preprocess_info


def print_label_distribution(labels, subset_name):
    """Print binary label distribution for diagnostics."""
    labels = np.asarray(labels)
    total = len(labels)
    pos = int(np.sum(labels == 1))
    neg = int(np.sum(labels == 0))
    pos_ratio = pos / total if total > 0 else 0.0
    neg_ratio = neg / total if total > 0 else 0.0
    print(
        f"  {subset_name}: {total} samples "
        f"(Pos: {pos}, Neg: {neg}, Pos ratio: {pos_ratio:.4f}, Neg ratio: {neg_ratio:.4f})"
    )


def _build_group_indices(groups):
    """Build mapping from group value to sample indices."""
    group_to_indices = defaultdict(list)
    for idx, g in enumerate(groups):
        group_to_indices[str(g).strip()].append(idx)
    return group_to_indices


def _score_group_split(test_n, test_pos, total_n, total_pos, test_size,
                       size_weight=1.0, label_weight=2.0, ratio_weight=1.0):
    """
    Score a candidate group split.

    Lower is better. The score jointly optimizes:
    1) test-set size close to test_size;
    2) positive/negative counts close to stratified targets;
    3) positive ratio close to the overall positive ratio.
    """
    total_neg = total_n - total_pos
    test_neg = test_n - test_pos

    target_test_n = total_n * test_size
    target_test_pos = total_pos * test_size
    target_test_neg = total_neg * test_size

    size_error = abs(test_n - target_test_n) / max(total_n, 1)

    pos_error = abs(test_pos - target_test_pos) / max(total_pos, 1)
    neg_error = abs(test_neg - target_test_neg) / max(total_neg, 1)
    label_error = pos_error + neg_error

    global_pos_ratio = total_pos / total_n if total_n > 0 else 0.0
    test_pos_ratio = test_pos / test_n if test_n > 0 else 0.0
    ratio_error = abs(test_pos_ratio - global_pos_ratio)

    # Strongly discourage invalid splits where train or test loses one class.
    penalty = 0.0
    train_pos = total_pos - test_pos
    train_neg = total_neg - test_neg
    if total_pos > 0 and (test_pos == 0 or train_pos == 0):
        penalty += 1000.0
    if total_neg > 0 and (test_neg == 0 or train_neg == 0):
        penalty += 1000.0

    return size_weight * size_error + label_weight * label_error + ratio_weight * ratio_error + penalty


def stratified_group_cold_start_split(groups, labels, test_size=0.2, random_state=42,
                                      n_trials=5000, candidates_per_step=64,
                                      size_weight=1.0, label_weight=2.0,
                                      ratio_weight=1.0):
    """
    Group-aware stratified train/test split for cold-start evaluation.

    This function assigns each group entirely to either train or test, avoiding
    ligand/protein leakage. Among many randomized greedy candidate splits, it
    selects the split whose test set is closest to the target test size and whose
    positive/negative label distribution is closest to the full dataset.

    Parameters
    ----------
    groups : array-like
        Group identifier for each sample. For ligand cold-start, use SMILES.
        For protein cold-start, use protein sequence.
    labels : array-like
        Binary labels, expected to contain 0 and 1.
    test_size : float
        Desired test-set proportion.
    random_state : int
        Random seed.
    n_trials : int
        Number of randomized trials. Larger values usually produce a better
        size/label-ratio match, at the cost of runtime.
    candidates_per_step : int
        Number of candidate groups considered at each greedy step.
    size_weight, label_weight, ratio_weight : float
        Weights for the objective function.

    Returns
    -------
    train_indices, test_indices : np.ndarray
        Sample indices for train and test.
    split_info : dict
        Diagnostics describing the selected split.
    """
    if not 0 < test_size < 1:
        raise ValueError(f"test_size must be between 0 and 1, got {test_size}")

    rng = np.random.default_rng(random_state)
    groups = np.asarray([str(g).strip() for g in groups], dtype=object)
    labels = np.asarray(labels)

    unique_labels = set(np.unique(labels).tolist())
    if not unique_labels.issubset({0, 1}):
        raise ValueError(
            f"stratified_group_cold_start_split expects binary labels 0/1, got {sorted(unique_labels)}"
        )

    total_n = len(labels)
    total_pos = int(np.sum(labels == 1))
    total_neg = total_n - total_pos

    if total_n == 0:
        raise ValueError("Cannot split an empty dataset.")

    if total_pos == 0 or total_neg == 0:
        raise ValueError(
            "Cannot perform label-stratified splitting because the dataset contains only one class."
        )

    group_to_indices = _build_group_indices(groups)
    unique_groups = np.array(list(group_to_indices.keys()), dtype=object)

    if len(unique_groups) < 2:
        raise ValueError(
            "Cannot perform cold-start split because fewer than two unique groups are available."
        )

    group_n = {}
    group_pos = {}
    for g, idxs in group_to_indices.items():
        idxs_arr = np.asarray(idxs, dtype=int)
        group_n[g] = len(idxs_arr)
        group_pos[g] = int(np.sum(labels[idxs_arr] == 1))

    target_test_n = total_n * test_size

    best_score = np.inf
    best_test_groups = None
    best_stats = None

    # Randomized greedy search. Each trial builds one group-exclusive test set.
    for trial in range(max(1, n_trials)):
        remaining = unique_groups.copy()
        rng.shuffle(remaining)
        remaining = list(remaining)

        test_groups = []
        test_group_set = set()
        test_n = 0
        test_pos = 0

        # Keep adding groups until the test-set target is reached or exceeded.
        # Candidate selection is randomized but scored, giving a practical
        # approximation to stratified group splitting.
        while remaining and test_n < target_test_n:
            if len(remaining) <= candidates_per_step:
                candidate_positions = np.arange(len(remaining))
            else:
                candidate_positions = rng.choice(
                    len(remaining), size=candidates_per_step, replace=False
                )

            best_candidate_pos = None
            best_candidate_score = np.inf

            for pos in candidate_positions:
                g = remaining[int(pos)]
                cand_n = test_n + group_n[g]
                cand_pos = test_pos + group_pos[g]
                cand_score = _score_group_split(
                    cand_n, cand_pos, total_n, total_pos, test_size,
                    size_weight=size_weight,
                    label_weight=label_weight,
                    ratio_weight=ratio_weight
                )

                if cand_score < best_candidate_score:
                    best_candidate_score = cand_score
                    best_candidate_pos = int(pos)

            selected_group = remaining.pop(best_candidate_pos)
            test_groups.append(selected_group)
            test_group_set.add(selected_group)
            test_n += group_n[selected_group]
            test_pos += group_pos[selected_group]

        score = _score_group_split(
            test_n, test_pos, total_n, total_pos, test_size,
            size_weight=size_weight,
            label_weight=label_weight,
            ratio_weight=ratio_weight
        )

        if score < best_score:
            best_score = score
            best_test_groups = set(test_groups)
            best_stats = {
                "score": float(score),
                "test_n": int(test_n),
                "test_pos": int(test_pos),
                "test_neg": int(test_n - test_pos),
                "test_group_count": int(len(test_groups)),
                "trial": int(trial),
            }

    if best_test_groups is None:
        raise RuntimeError("Failed to construct a stratified group cold-start split.")

    test_indices = []
    train_indices = []

    for g, idxs in group_to_indices.items():
        if g in best_test_groups:
            test_indices.extend(idxs)
        else:
            train_indices.extend(idxs)

    train_indices = np.asarray(train_indices, dtype=int)
    test_indices = np.asarray(test_indices, dtype=int)

    rng.shuffle(train_indices)
    rng.shuffle(test_indices)

    train_groups = set(groups[train_indices].tolist())
    test_groups = set(groups[test_indices].tolist())
    overlap = train_groups.intersection(test_groups)

    if overlap:
        raise RuntimeError(
            f"Group leakage detected: {len(overlap)} groups appear in both train and test."
        )

    split_info = {
        **best_stats,
        "train_n": int(len(train_indices)),
        "train_pos": int(np.sum(labels[train_indices] == 1)),
        "train_neg": int(np.sum(labels[train_indices] == 0)),
        "train_group_count": int(len(train_groups)),
        "total_group_count": int(len(unique_groups)),
        "target_test_size": float(test_size),
        "actual_test_size": float(len(test_indices) / total_n),
        "global_pos_ratio": float(total_pos / total_n),
        "train_pos_ratio": float(np.sum(labels[train_indices] == 1) / max(len(train_indices), 1)),
        "test_pos_ratio": float(np.sum(labels[test_indices] == 1) / max(len(test_indices), 1)),
    }

    return train_indices, test_indices, split_info


def balance_binary_subset(smiles, proteins, labels, descriptors, random_state=42, subset_name="subset"):
    """
    Balance one subset to a 1:1 positive/negative ratio by random undersampling.

    This is applied after splitting. For cold-start splits, it does not introduce
    group leakage because samples are removed only within the already separated
    train/test subsets.
    """
    rng = np.random.default_rng(random_state)

    labels = np.asarray(labels)
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]

    if len(pos_idx) == 0 or len(neg_idx) == 0:
        raise ValueError(
            f"Cannot balance {subset_name}: Pos={len(pos_idx)}, Neg={len(neg_idx)}. "
            "Both classes must be present."
        )

    min_count = min(len(pos_idx), len(neg_idx))

    pos_sampled = rng.choice(pos_idx, min_count, replace=False)
    neg_sampled = rng.choice(neg_idx, min_count, replace=False)
    balanced_idx = np.concatenate([pos_sampled, neg_sampled])
    rng.shuffle(balanced_idx)

    return (
        smiles[balanced_idx],
        proteins[balanced_idx],
        labels[balanced_idx],
        descriptors[balanced_idx],
    )


def process_data(file_path, max_protein_len=480, test_size=0.2, random_state=42,
                 ligand_cold_start=False, protein_cold_start=False, balance_samples=False,
                 variance_threshold=0.001, vif_threshold=10.0, max_iterations=150, min_features=50,
                 stratified_group_trials=5000):

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
    
    print("\n[4/7] Splitting train/test sets with configured strategies...")
    # ================= 冷启动/随机划分 核心逻辑 =================
    smiles_arr = np.array(smiles_list, dtype=object)
    protein_arr = np.array(protein_list, dtype=object)
    labels_arr = np.array(labels)
    desc_arr = full_descriptors
    
    if ligand_cold_start:
        print("  → Strategy: Stratified Group Ligand Cold Start (配体分组分层冷启动)")
        train_indices, test_indices, split_info = stratified_group_cold_start_split(
            groups=smiles_arr,
            labels=labels_arr,
            test_size=test_size,
            random_state=random_state,
            n_trials=stratified_group_trials
        )
        print("  Stratified group split diagnostics:")
        print(f"    Total ligand groups: {split_info['total_group_count']}")
        print(f"    Train ligand groups: {split_info['train_group_count']}")
        print(f"    Test ligand groups: {split_info['test_group_count']}")
        print(f"    Target test size: {split_info['target_test_size']:.4f}")
        print(f"    Actual test size: {split_info['actual_test_size']:.4f}")
        print(f"    Global pos ratio: {split_info['global_pos_ratio']:.4f}")
        print(f"    Train pos ratio: {split_info['train_pos_ratio']:.4f}")
        print(f"    Test pos ratio: {split_info['test_pos_ratio']:.4f}")
        print(f"    Best split score: {split_info['score']:.6f} (trial {split_info['trial']})")

    elif protein_cold_start:
        print("  → Strategy: Stratified Group Protein Cold Start (蛋白分组分层冷启动)")
        train_indices, test_indices, split_info = stratified_group_cold_start_split(
            groups=protein_arr,
            labels=labels_arr,
            test_size=test_size,
            random_state=random_state,
            n_trials=stratified_group_trials
        )
        print("  Stratified group split diagnostics:")
        print(f"    Total protein groups: {split_info['total_group_count']}")
        print(f"    Train protein groups: {split_info['train_group_count']}")
        print(f"    Test protein groups: {split_info['test_group_count']}")
        print(f"    Target test size: {split_info['target_test_size']:.4f}")
        print(f"    Actual test size: {split_info['actual_test_size']:.4f}")
        print(f"    Global pos ratio: {split_info['global_pos_ratio']:.4f}")
        print(f"    Train pos ratio: {split_info['train_pos_ratio']:.4f}")
        print(f"    Test pos ratio: {split_info['test_pos_ratio']:.4f}")
        print(f"    Best split score: {split_info['score']:.6f} (trial {split_info['trial']})")

    else:
        print("  → Strategy: Random Stratified (随机分层划分)")
        indices = np.arange(len(smiles_list))
        train_indices, test_indices = train_test_split(
            indices, test_size=test_size, random_state=random_state, stratify=labels_arr
        )

    # 提取初始划分的数据
    smiles_train = smiles_arr[train_indices]
    protein_train = protein_arr[train_indices]
    labels_train = labels_arr[train_indices]
    train_descriptors = desc_arr[train_indices]
    
    smiles_test = smiles_arr[test_indices]
    protein_test = protein_arr[test_indices]
    labels_test = labels_arr[test_indices]
    test_descriptors = desc_arr[test_indices]

    print("  Initial split label distribution:")
    print_label_distribution(labels_train, "Train before balancing")
    print_label_distribution(labels_test, "Test before balancing")

    # ================= 切分后的集合内动态强制平衡 (1:1) =================
    if balance_samples:
        print("  → Strategy: Post-split Dynamic Balancing by Random Undersampling (严格1:1平衡)")
        smiles_train, protein_train, labels_train, train_descriptors = balance_binary_subset(
            smiles_train, protein_train, labels_train, train_descriptors,
            random_state=random_state, subset_name="training set"
        )

        smiles_test, protein_test, labels_test, test_descriptors = balance_binary_subset(
            smiles_test, protein_test, labels_test, test_descriptors,
            random_state=random_state + 1, subset_name="test set"
        )

        print("  Balanced label distribution:")
        print_label_distribution(labels_train, "Train after balancing")
        print_label_distribution(labels_test, "Test after balancing")

    print(f"  Train set: {len(smiles_train)} samples")
    print(f"  Test set: {len(smiles_test)} samples")
    print(f"  ✓ Split completed")

    print("\n[5/7] Applying Variance + VIF filtering on training set...")
    
    desc_df = pd.DataFrame(full_descriptors, columns=all_descriptor_names)
    desc_df.insert(0, 'SMILES', smiles_list)
    desc_df.to_csv('descriptors/all_descriptors_raw.csv', index=False)
    print(f"  Saving all descriptors to: descriptors/all_descriptors_raw.csv")
    
    train_desc_df = pd.DataFrame(train_descriptors, columns=all_descriptor_names)
    train_desc_df.insert(0, 'SMILES', smiles_train)
    train_desc_df.to_csv('descriptors/train_descriptors_raw.csv', index=False)
    
    filtered_descriptors_train, feature_indices, feature_names, descriptor_preprocess_info = filter_descriptors_by_variance_and_vif(
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
        'vif_threshold': vif_threshold,
        'descriptor_preprocessing': descriptor_preprocess_info
    }
    with open('descriptors/descriptor_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"  {len(feature_names)} descriptors retained after filtering")
    
    print("\n[6/7] Saving filtered descriptors...")
    # Use the same training-set cleaning/imputation and selected descriptor names
    # for both train and test. Do not slice raw arrays directly, because selected
    # columns may still contain NaN values in raw RDKit output.
    train_desc_filtered = apply_descriptor_preprocessing(
        train_descriptors, all_descriptor_names, feature_names, descriptor_preprocess_info
    )
    test_desc_filtered = apply_descriptor_preprocessing(
        test_descriptors, all_descriptor_names, feature_names, descriptor_preprocess_info
    )

    if train_desc_filtered.shape != filtered_descriptors_train.shape:
        raise RuntimeError(
            f"Training descriptor shape mismatch: transformed={train_desc_filtered.shape}, "
            f"filtered={filtered_descriptors_train.shape}"
        )

    print(f"  Final train descriptor matrix: {train_desc_filtered.shape}")
    print(f"  Final test descriptor matrix: {test_desc_filtered.shape}")
    
    print("\n[7/7] Processing protein sequences and saving final data...")
    os.makedirs('models', exist_ok=True)
    with open('models/protein_tokenizer_chars.txt', 'w') as f:
        f.write(','.join(protein_chars))

    # 训练集蛋白补齐
    processed_proteins_train = []
    for protein in protein_train:
        if len(protein) > max_protein_len:
            processed = protein[:max_protein_len]
        else:
            processed = protein.ljust(max_protein_len)
        processed_proteins_train.append(processed)
        
    # 测试集蛋白补齐
    processed_proteins_test = []
    for protein in protein_test:
        if len(protein) > max_protein_len:
            processed = protein[:max_protein_len]
        else:
            processed = protein.ljust(max_protein_len)
        processed_proteins_test.append(processed)

    os.makedirs('processed_data', exist_ok=True)
    
    print(f"  Saving training set data...")
    np.savez('processed_data/train.npz', 
             smiles=smiles_train, 
             proteins=processed_proteins_train, 
             labels=labels_train,
             descriptors=train_desc_filtered,
             allow_pickle=True)
    
    print(f"  Saving test set data...")
    np.savez('processed_data/test.npz', 
             smiles=smiles_test, 
             proteins=processed_proteins_test, 
             labels=labels_test,
             descriptors=test_desc_filtered,
             allow_pickle=True)

    with open('processed_data/train.txt', 'w') as f_train:
        for s, p, l in zip(smiles_train, processed_proteins_train, labels_train):
            f_train.write(f"{s} {p.strip()} {l}\n")
    
    with open('processed_data/test.txt', 'w') as f_test:
        for s, p, l in zip(smiles_test, processed_proteins_test, labels_test):
            f_test.write(f"{s} {p.strip()} {l}\n")
            
    total_time = time.time() - start_time
    
    print("\n" + "="*70)
    print("Data preprocessing completed!")
    print(f"Total time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
    print("="*70)
    print(f"Training set: {len(smiles_train)} samples (Pos: {np.sum(labels_train==1)}, Neg: {np.sum(labels_train==0)})")
    print(f"Test set: {len(smiles_test)} samples (Pos: {np.sum(labels_test==1)}, Neg: {np.sum(labels_test==0)})")
    
    return {
        'train': {'smiles': smiles_train, 'proteins': processed_proteins_train, 'labels': labels_train},
        'test': {'smiles': smiles_test, 'proteins': processed_proteins_test, 'labels': labels_test},
        'actual_max_smiles': actual_max_smiles,
        'actual_max_protein': actual_max_protein,
        'filtered_descriptor_names': feature_names,
        'descriptor_preprocessing': descriptor_preprocess_info
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process compound-protein interaction data.")
    parser.add_argument('--file_path', type=str, default='dataset.txt', help='Path to dataset.')
    parser.add_argument('--ligand_cold_start', action='store_true', help='Enable ligand cold start.')
    parser.add_argument('--protein_cold_start', action='store_true', help='Enable protein cold start.')
    parser.add_argument('--balance_samples', action='store_true', help='Force equal number of positive/negative samples.')
    parser.add_argument('--max_protein_len', type=int, default=480, help='Max protein length.')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test size ratio.')
    parser.add_argument('--variance_threshold', type=float, default=0.001, help='Variance threshold.')
    parser.add_argument('--vif_threshold', type=float, default=10.0, help='VIF threshold.')
    parser.add_argument('--max_iterations', type=int, default=150, help='Max VIF iterations.')
    parser.add_argument('--min_features', type=int, default=50, help='Min retained features.')
    parser.add_argument('--random_state', type=int, default=42, help='Random state.')
    parser.add_argument('--stratified_group_trials', type=int, default=5000,
                        help='Number of randomized trials for stratified group cold-start split.')
    args = parser.parse_args()

    if args.ligand_cold_start and args.protein_cold_start:
        raise ValueError("Cannot enable both ligand cold start and protein cold start at the same time. Please choose one.")

    print("\n" + "="*70)
    print("Configuration parameters")
    print("="*70)
    print(f"Dataset file: {args.file_path}")
    print(f"Split Mode: {'Stratified Group Ligand Cold Start' if args.ligand_cold_start else 'Stratified Group Protein Cold Start' if args.protein_cold_start else 'Random Stratified'}")
    print(f"Balance Samples: {args.balance_samples}")
    print(f"Stratified Group Trials: {args.stratified_group_trials}")
    
    processed_data = process_data(
        file_path=args.file_path,
        max_protein_len=args.max_protein_len,
        test_size=args.test_size,
        random_state=args.random_state,
        ligand_cold_start=args.ligand_cold_start,
        protein_cold_start=args.protein_cold_start,
        balance_samples=args.balance_samples,
        variance_threshold=args.variance_threshold,
        vif_threshold=args.vif_threshold,
        max_iterations=args.max_iterations,
        min_features=args.min_features,
        stratified_group_trials=args.stratified_group_trials
    )