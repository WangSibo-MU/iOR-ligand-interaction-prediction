# enhanced_qsar_cv.py
# -*- coding: utf-8 -*-
"""
Streamlined QSAR with 5-Fold Cross-Validation and Independent Test Set
"""

# ==================== Windows UTF-8 Handling ====================
import os
import sys
if os.name == 'nt':
    if not (sys.flags.utf8_mode == 1 or os.environ.get('PYTHONUTF8') == '1'):
        env = os.environ.copy()
        env.update({'PYTHONUTF8': '1', 'PYTHONIOENCODING': 'utf-8'})
        os.execve(sys.executable, [sys.executable, '-X', 'utf8'] + sys.argv, env)

import pandas as pd
import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')

from rdkit import Chem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score, matthews_corrcoef, 
    recall_score, confusion_matrix
)

# ==================== Configuration ====================
output_dir = 'enhanced_cv_results'
os.makedirs(output_dir, exist_ok=True)

# ==================== 1. Data Loading ====================
print("Loading training data...")
train_df = pd.read_csv('train.txt', sep=' ', header=None, names=['smiles', 'protein', 'label'])
train_df = train_df[['smiles', 'label']]

print("\nLoading test data...")
test_df = pd.read_csv('test.txt', sep=' ', header=None, names=['smiles', 'protein', 'label'])
test_df = test_df[['smiles', 'label']]

# Check class distribution
print("\nTraining Set Class Distribution:")
print(train_df['label'].value_counts())
print("\nTest Set Class Distribution:")
print(test_df['label'].value_counts())

# Remove invalid SMILES
def validate_smiles(df, name):
    valid_mask = df['smiles'].apply(lambda x: Chem.MolFromSmiles(x) is not None)
    valid_df = df[valid_mask].copy()
    print(f"{name} valid data: {len(valid_df)} records (removed {len(df) - len(valid_df)})")
    return valid_df

train_df = validate_smiles(train_df, "Training")
test_df = validate_smiles(test_df, "Test")

# ==================== 2. Feature Engineering ====================
def smiles_to_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(2048, dtype=int)
    generator = GetMorganGenerator(radius=2, fpSize=2048)
    return np.array(generator.GetFingerprintAsNumPy(mol), dtype=int)

print("\nGenerating fingerprints for training data...")
X_train_full = pd.DataFrame(np.vstack(train_df['smiles'].apply(smiles_to_fingerprint)), 
                            columns=[f'fp_{i}' for i in range(2048)])
y_train_full = train_df['label'].values

print("Generating fingerprints for test data...")
X_test = pd.DataFrame(np.vstack(test_df['smiles'].apply(smiles_to_fingerprint)), 
                      columns=[f'fp_{i}' for i in range(2048)])
y_test = test_df['label'].values

print(f"\nFinal shapes - Training: {X_train_full.shape}, Test: {X_test.shape}")

# ==================== 3. Model Definitions ====================
models = {
    'RandomForest': RandomForestClassifier(
        n_estimators=300, max_depth=None, min_samples_split=2,
        min_samples_leaf=1, max_features='sqrt', class_weight='balanced',
        random_state=42, n_jobs=-1
    ),
    'SVM': SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42),
    'GradientBoosting': GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.1, max_depth=6, random_state=42
    ),
    'XGBoost': None
}

try:
    from xgboost import XGBClassifier
    models['XGBoost'] = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1, 
        subsample=0.8, random_state=42, n_jobs=-1, eval_metric='logloss'
    )
    print("XGBoost imported successfully")
except ImportError:
    print("Warning: XGBoost not available")
    del models['XGBoost']

# ==================== 4. Training and Evaluation Functions ====================
def train_and_evaluate_cv(model_name, model, X, y, X_test, y_test, output_base_dir):
    print(f"\n{'='*50}")
    print(f"Training {model_name} with 5-Fold Cross-Validation...")
    print(f"{'='*50}")
    
    model_dir = os.path.join(output_base_dir, model_name.lower())
    os.makedirs(model_dir, exist_ok=True)
    
    # 5-Fold Stratified Cross-Validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = []
    fold = 1
    
    for train_idx, val_idx in skf.split(X, y):
        print(f"\n--- Fold {fold} ---")
        
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        # Train model for this fold
        start_time = time.time()
        model_fold = model.__class__(**model.get_params())
        model_fold.fit(X_train_fold, y_train_fold)
        training_time = time.time() - start_time
        
        # Validate on validation set
        y_val_pred = model_fold.predict(X_val_fold)
        y_val_proba = model_fold.predict_proba(X_val_fold)[:, 1] if hasattr(model_fold, 'predict_proba') else None
        
        # Calculate metrics for this fold
        fold_metrics = {
            'Fold': fold,
            'Accuracy': accuracy_score(y_val_fold, y_val_pred),
            'F1_Score': f1_score(y_val_fold, y_val_pred),
            'Recall': recall_score(y_val_fold, y_val_pred),
            'MCC': matthews_corrcoef(y_val_fold, y_val_pred),
            'Training_Time': training_time
        }
        if y_val_proba is not None:
            fold_metrics['AUC'] = roc_auc_score(y_val_fold, y_val_proba)
        
        cv_results.append(fold_metrics)
        fold += 1
    
    # Calculate CV statistics
    cv_df = pd.DataFrame(cv_results).set_index('Fold')
    cv_stats = cv_df.agg(['mean', 'std'])
    
    print(f"\n--- {model_name} Cross-Validation Results ---")
    print(cv_df.to_string())
    print("\n--- Cross-Validation Statistics ---")
    for metric in ['Accuracy', 'F1_Score', 'Recall', 'MCC', 'AUC']:
        if metric in cv_df.columns:
            mean_val = cv_stats.loc['mean', metric]
            std_val = cv_stats.loc['std', metric]
            print(f"{metric}: {mean_val:.4f} ± {std_val:.4f}")
    
    # Save CV results
    cv_path = os.path.join(model_dir, 'cv_results.csv')
    cv_df.to_csv(cv_path, encoding='utf-8-sig')
    print(f"\nCross-validation results saved to: {cv_path}")
    
    # Train final model on full training set
    print(f"\nTraining final {model_name} on full training set...")
    start_time = time.time()
    final_model = model.__class__(**model.get_params())
    final_model.fit(X, y)
    final_training_time = time.time() - start_time
    
    # Evaluate on independent test set
    print(f"\nEvaluating {model_name} on independent test set...")
    y_test_pred = final_model.predict(X_test)
    y_test_proba = final_model.predict_proba(X_test)[:, 1] if hasattr(final_model, 'predict_proba') else None
    
    # Test set metrics
    test_metrics = {
        'Accuracy': accuracy_score(y_test, y_test_pred),
        'F1_Score': f1_score(y_test, y_test_pred),
        'Recall': recall_score(y_test, y_test_pred),
        'MCC': matthews_corrcoef(y_test, y_test_pred),
        'Training_Time': final_training_time
    }
    if y_test_proba is not None:
        test_metrics['AUC'] = roc_auc_score(y_test, y_test_proba)
    
    # Confusion Matrix for test set
    cm = confusion_matrix(y_test, y_test_pred)
    print(f"\nTest Set Confusion Matrix:\n{cm}")
    
    # Print test metrics
    print(f"\n--- {model_name} Test Set Performance ---")
    for name, value in test_metrics.items():
        print(f"{name}: {value:.4f}")
    
    # Save test results
    metrics_path = os.path.join(model_dir, 'test_metrics.txt')
    with open(metrics_path, 'w', encoding='utf-8', errors='ignore') as f:
        f.write(f"{model_name} Model - Independent Test Set Performance\n" + "="*50 + "\n\n")
        for name, value in test_metrics.items():
            f.write(f"{name}: {value:.4f}\n")
        f.write(f"\nConfusion Matrix:\n{cm}\n")
        f.write("\nCross-Validation Statistics:\n")
        for metric in ['Accuracy', 'F1_Score', 'Recall', 'MCC', 'AUC']:
            if metric in cv_df.columns:
                mean_val = cv_stats.loc['mean', metric]
                std_val = cv_stats.loc['std', metric]
                f.write(f"{metric}: {mean_val:.4f} ± {std_val:.4f}\n")
    print(f"\nTest metrics saved to: {metrics_path}")
    
    # Feature importance (from final model)
    if hasattr(final_model, 'feature_importances_'):
        importances = final_model.feature_importances_
        top_indices = np.argsort(importances)[-20:][::-1]
        
        importance_path = os.path.join(model_dir, 'feature_importance.txt')
        with open(importance_path, 'w', encoding='utf-8', errors='ignore') as f:
            f.write(f"Top 20 Important Fingerprint Bits ({model_name})\n" + "="*40 + "\n\n")
            f.write("Rank\tBit_Index\tImportance\n")
            for rank, idx in enumerate(top_indices, 1):
                f.write(f"{rank}\t{idx}\t{importances[idx]:.6f}\n")
        print(f"Feature importance saved to: {importance_path}")
    
    # Save final model
    try:
        import joblib
        joblib.dump(final_model, os.path.join(model_dir, f'{model_name.lower()}_final_model.pkl'))
        print(f"Final model saved to: {model_dir}")
    except ImportError:
        print("joblib not installed, model not saved")
    
    return {
        'cv_results': cv_df,
        'cv_stats': cv_stats,
        'test_metrics': test_metrics,
        'confusion_matrix': cm
    }

# ==================== 5. Train All Models ====================
print("\nStarting model training pipeline with 5-Fold Cross-Validation...")
results = {}

for name, model in models.items():
    if model is not None:
        results[name] = train_and_evaluate_cv(
            name, model, X_train_full, y_train_full, X_test, y_test, output_dir
        )

# ==================== 6. Summary ====================
print("\n" + "="*60)
print("FINAL PERFORMANCE SUMMARY")
print("="*60)

# Cross-validation summary
print("\nCross-Validation Performance (Mean ± Std):")
cv_summary_data = []
for name, result in results.items():
    if result is not None:
        cv_stats = result['cv_stats']
        row = {'Model': name}
        for metric in ['Accuracy', 'F1_Score', 'Recall', 'MCC', 'AUC']:
            if metric in cv_stats.columns:
                row[f'{metric}_Mean'] = cv_stats.loc['mean', metric]
                row[f'{metric}_Std'] = cv_stats.loc['std', metric]
        cv_summary_data.append(row)

if cv_summary_data:
    cv_summary_df = pd.DataFrame(cv_summary_data).set_index('Model')
    print("\nCV Performance:")
    print(cv_summary_df.to_string())
    
    cv_summary_path = os.path.join(output_dir, 'cv_summary.csv')
    cv_summary_df.to_csv(cv_summary_path, encoding='utf-8-sig')
    print(f"\nCV summary saved to: {cv_summary_path}")

# Independent test set summary
print("\n" + "="*60)
print("INDEPENDENT TEST SET PERFORMANCE")
print("="*60)

test_summary_data = []
for name, result in results.items():
    if result is not None:
        test_metrics = result['test_metrics']
        row = {'Model': name}
        row.update(test_metrics)
        test_summary_data.append(row)

if test_summary_data:
    test_summary_df = pd.DataFrame(test_summary_data).set_index('Model')
    print("\nTest Set Performance:")
    print(test_summary_df.to_string())
    
    test_summary_path = os.path.join(output_dir, 'test_summary.csv')
    test_summary_df.to_csv(test_summary_path, encoding='utf-8-sig')
    print(f"\nTest summary saved to: {test_summary_path}")

print("\nAll results saved to folder:", output_dir)