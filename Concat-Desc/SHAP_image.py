import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats
import shap

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Arial'
plt.rcParams['mathtext.it'] = 'Arial:italic'
plt.rcParams['mathtext.bf'] = 'Arial:bold'
plt.rcParams['axes.unicode_minus'] = False

RESULTS_DIR = "shap_analysis_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_shap_data(file_path):
    print(f"Loading SHAP data from {file_path}...")
    df = pd.read_csv(file_path)
    print(f"Data loaded successfully. Shape: {df.shape}")
    return df

def load_feature_data(file_path):
    print(f"Loading feature data from {file_path}...")
    df = pd.read_csv(file_path)
    print(f"Feature data loaded successfully. Shape: {df.shape}")
    return df

def identify_shap_columns(df):
    meta_columns = ['Sample_Index', 'SMILES', 'Protein_Sequence', 'True_Label', 'Base_Value']
    existing_meta_cols = [col for col in meta_columns if col in df.columns]
    shap_columns = [col for col in df.columns if col not in existing_meta_cols]
    
    print(f"Found {len(shap_columns)} SHAP feature columns")
    return shap_columns, existing_meta_cols

def identify_feature_columns(df):
    meta_columns = ['Sample_Index', 'SMILES', 'Protein_Sequence', 'True_Label', 'Base_Value']
    existing_meta_cols = [col for col in meta_columns if col in df.columns]
    feature_columns = [col for col in df.columns if col not in existing_meta_cols]
    
    print(f"Found {len(feature_columns)} feature columns")
    return feature_columns, existing_meta_cols

def plot_shap_summary(shap_values, feature_values, feature_names, max_display=20):
    print("Creating SHAP summary plot...")
    plt.rcParams['font.family'] = 'Arial'
    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values, 
                     features=feature_values,
                     feature_names=feature_names, 
                     max_display=max_display, 
                     show=False)
    plt.title(f'SHAP Summary Plot (Top {max_display} Features)', fontsize=16, fontname='Arial')
    for item in ([plt.gca().title, plt.gca().xaxis.label, plt.gca().yaxis.label] +
                 plt.gca().get_xticklabels() + plt.gca().get_yticklabels()):
        item.set_fontname('Arial')
        item.set_fontsize(10 if hasattr(item, 'get_text') and item.get_text() else 12)
    
    plt.tight_layout()
    output_path = os.path.join(RESULTS_DIR, 'shap_summary_plot.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"SHAP summary plot saved to {output_path}")

def plot_feature_importance(df, shap_columns, top_n=20):
    print("Creating feature importance bar plot...")
    mean_abs_shap = df[shap_columns].abs().mean().sort_values(ascending=True)
    top_features = mean_abs_shap.tail(top_n)

    plt.figure(figsize=(12, 10))
    bars = plt.barh(range(len(top_features)), top_features.values, alpha=0.7, color='steelblue')
    plt.yticks(range(len(top_features)), top_features.index, fontsize=10, fontname='Arial')
    plt.xlabel('Mean |SHAP Value| (Average Impact on Model Output)', fontsize=12, fontname='Arial')
    plt.title(f'Top {top_n} Feature Importance (Based on SHAP Values)', fontsize=14, fontname='Arial')
    plt.grid(True, alpha=0.3, axis='x')

    plt.xticks(fontname='Arial')
    
    plt.tight_layout()

    output_path = os.path.join(RESULTS_DIR, 'shap_feature_importance.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Feature importance plot saved to {output_path}")
    return top_features.index.tolist()

def plot_feature_correlation(df, shap_columns, top_n=20):
    print("Creating feature correlation heatmap...")

    mean_abs_shap = df[shap_columns].abs().mean().sort_values(ascending=False)

    top_features = mean_abs_shap.head(top_n).index.tolist()

    correlation_matrix = df[top_features].corr()

    plt.figure(figsize=(14, 12))
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

    sns.heatmap(correlation_matrix, 
                mask=mask,
                cmap='RdBu_r', 
                center=0,
                square=True, 
                annot=False,
                cbar_kws={"shrink": .8})
    
    plt.title(f'Feature Correlation Heatmap (Top {top_n} Features)', fontsize=16, fontname='Arial')

    plt.xticks(rotation=45, ha='right', fontname='Arial')
    plt.yticks(rotation=0, fontname='Arial')

    cbar = plt.gca().collections[0].colorbar
    cbar.ax.tick_params(labelsize=10)
    for label in cbar.ax.get_yticklabels():
        label.set_fontname('Arial')
    
    plt.tight_layout()

    output_path = os.path.join(RESULTS_DIR, 'shap_feature_correlation.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Feature correlation heatmap saved to {output_path}")

def analyze_by_class(df, shap_columns, meta_columns):
    if 'True_Label' not in meta_columns:
        print("No True_Label column found. Skipping class-based analysis.")
        return
    
    print("Performing class-based analysis...")

    classes = df['True_Label'].unique()
    if len(classes) < 2:
        print("Only one class found. Skipping class-based analysis.")
        return

    class_shap_means = {}
    for cls in classes:
        class_data = df[df['True_Label'] == cls]
        class_shap_means[cls] = class_data[shap_columns].mean()

    comparison_df = pd.DataFrame(class_shap_means)

    mean_abs_shap = df[shap_columns].abs().mean().sort_values(ascending=False)
    top_features = mean_abs_shap.head(10).index.tolist()

    plt.figure(figsize=(12, 8))
    comparison_df.loc[top_features].plot(kind='bar', figsize=(12, 8))
    plt.title('Mean SHAP Values by Class (Top 10 Features)', fontname='Arial')
    plt.ylabel('Mean SHAP Value', fontname='Arial')
    plt.xlabel('Feature', fontname='Arial')
    plt.xticks(rotation=45, fontname='Arial')
    plt.yticks(fontname='Arial')
    plt.legend(title='Class', prop={'family': 'Arial'})
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path = os.path.join(RESULTS_DIR, 'shap_class_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Class comparison plot saved to {output_path}")

def main():
    shap_file_path = "SHAP/all_samples_shap_values.csv"  # SHAP value file path
    feature_file_path = "descriptors/all_samples_data.csv"  # Feature value file path
    
    df_shap = load_shap_data(shap_file_path)
    df_features = load_feature_data(feature_file_path)

    shap_columns, meta_columns = identify_shap_columns(df_shap)
    feature_columns, _ = identify_feature_columns(df_features)

    if set(shap_columns) != set(feature_columns):
        print("Warning: SHAP columns and feature columns do not match. Using SHAP column names.")
        df_features = df_features[shap_columns]
        feature_columns = shap_columns

    shap_values = df_shap[shap_columns].values
    feature_values = df_features[feature_columns].values
    plot_shap_summary(shap_values, feature_values, shap_columns)
    top_features_importance = plot_feature_importance(df_shap, shap_columns)
    plot_feature_correlation(df_shap, shap_columns)
    analyze_by_class(df_shap, shap_columns, meta_columns)
    mean_abs_shap = df_shap[shap_columns].abs().mean().sort_values(ascending=False)
    importance_df = pd.DataFrame({
        'Feature': mean_abs_shap.index,
        'Mean_Abs_SHAP': mean_abs_shap.values
    })
    importance_df.to_csv(os.path.join(RESULTS_DIR, 'feature_importance_ranking.csv'), index=False)
    
    print("\nAll analyses completed! Results saved to:", RESULTS_DIR)

if __name__ == '__main__':
    main()