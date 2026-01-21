### Insect Odorant Receptor (OR)-Ligand Interaction Prediction Model

This repository implements a comprehensive deep learning framework for predicting interactions between insect olfactory receptors (ORs) and odorant molecules (ligands). The system supports three complementary molecular representation approaches, each with full interpretability analysis capabilities.



**Project Overview**

The model predicts whether a given insect OR protein interacts with a specific odorant molecule. Three deep learning architectures are provided:

Concat-Desc: Traditional RDKit molecular descriptors combined with a protein sequence Transformer

Concat-GNN: Learning from molecular graph structures

Cross-Attn: Pure sequence-based modeling treating SMILES and proteins as text

All architectures include extensive explainability tools: SHAP analysis, attention visualization, and GNNExplainer.



**Environment Requirements**

torch=2.6.0+cu124

torch-geometric=2.6.1

rdkit=2024.9.6

numpy=1.26.4

pandas=2.3.3

scikit-learn=1.7.2

shap=0.48.0

matplotlib=3.10.0

seaborn=0.13.2



**Data Preparation**

SMILES\_STRING PROTEIN\_SEQUENCE LABEL

Example:

CCOCC1=CC=CC=C1 MKVMILFSSVILLLLVGILFSIGAVTLVNLGYATTMFGYITSPVL 1

C1=CC=C(C=C1)C=O MKVMILFSSVILLLLVGILFSIGAVTLVNLGYATTMFGYITSPVL 0



**Usage Workflow**

Choose one of the three approaches based on your needs:

1. Molecular Descriptors + Protein Transformer
2. Graph Neural Network + Protein Transformer
3. Both character-level Transformer

Step 1: Data Preprocessing

python data\_processing.py

Step 2: Model Training

python training.py

Step 3: Validation/Prediction

\# Set TASK = 'validate' or 'predict'

python validation\_prediction.py
