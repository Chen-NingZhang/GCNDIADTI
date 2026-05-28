# GCNDIADTI

## Title

GCNDIADTI: Graph Convolutional Networks and Deep Interactive Attention Module for Drug-Target Interaction Prediction

## Description

This README describes the code and data files used to reproduce the GCNDIADTI experiments. GCNDIADTI is a drug-target interaction prediction model that integrates multi-source drug similarity information, multi-source target similarity information, known drug-target associations, graph convolutional representation learning, and a deep interactive attention module.

## Dataset Information

The main data used in this study were based on the DTINet/Luo benchmark dataset and were used in the preprocessed form adopted by MIDTI. The original DTINet/Luo dataset was introduced by Luo et al. (2017), and the MIDTI preprocessing source was described by Song et al. (2024).

Original DTINet/Luo data source:

Luo Y, Zhao X, Zhou J, Yang J, Zhang Y, Kuang W, Peng J, Chen L, Zeng J. 2017. A network integration approach for drug-target interaction prediction and computational drug repositioning from heterogeneous information. Nature Communications 8:573. DOI: 10.1038/s41467-017-00680-8.

DTINet repository: https://github.com/luoyunan/DTINet

MIDTI preprocessing source:

Song W, Xu L, Han C, Tian Z. 2024. Drug-target interaction predictions with multi-view similarity network fusion strategy and deep interactive attention mechanism. Bioinformatics 40:btae346. DOI: 10.1093/bioinformatics/btae346.

MIDTI repository: https://github.com/XuLew/MIDTI

According to the manuscript, the Luo benchmark used in this study contains 1,923 known drug-target interactions between 708 drugs and 1,512 targets.

The manuscript also uses the Yamanishi et al. (2008) benchmark dataset for additional generalization experiments, specifically the enzyme and ion channel subsets.

Yamanishi data source:

Yamanishi Y, Araki M, Gutteridge A, Honda W, Kanehisa M. 2008. Prediction of drug-target interaction networks from the integration of chemical and genomic spaces. Bioinformatics 24:i232-i240. DOI: 10.1093/bioinformatics/btn162.

The input files required by data_process_5.py for the Luo benchmark are placed under dataset/LuoDTI/data/:

protein_drug_interaction.txt
drug_drug_interaction.csv
drug_disease_association.csv
drug_side_effect_association.csv
drug_chemical_structure.csv
protein_protein_interaction.csv
protein_disease_association.csv
protein_genome_sequence.csv

Brief file descriptions:

protein_drug_interaction.txt: known protein-drug interaction matrix used to construct positive DTI samples.
drug_drug_interaction.csv: drug-drug interaction similarity information.
drug_disease_association.csv: drug-disease association information.
drug_side_effect_association.csv: drug-side-effect association information.
drug_chemical_structure.csv: drug chemical-structure similarity information.
protein_protein_interaction.csv: protein-protein interaction similarity information.
protein_disease_association.csv: protein-disease association information.
protein_genome_sequence.csv: protein sequence or genomic similarity information.

## Code Information

The code package contains the following scripts:

data_process_5.py: prepares the fold-specific input tensors for five repeated 5-fold cross-validation experiments.
train_5.py: trains and evaluates GCNDIADTI and reports ACC, ROC-AUC, and AUPR.
model_all.py: defines the complete GCNDIADTI model.
Model517.py: defines graph convolution, feature integration, and classifier modules.
Transformer.py: defines the deep interactive attention module.
early_stopping.py: implements early stopping and checkpoint saving.
parameters.py: stores default model parameters and device settings.

Running the scripts generates the following outputs:

embed_index_adj_protein_drug_1to1_strict_repeat_0.pth to embed_index_adj_protein_drug_1to1_strict_repeat_4.pth: preprocessed fold-specific tensors.
best_parameter/: saved model checkpoints.
result/: saved prediction outputs.

## Requirements

The code requires Python and the following Python packages:

Python 3.7 or later
torch
numpy
pandas
scikit-learn
einops

## Usage Instructions

Run the scripts from the code root directory in the following order:

1. Prepare the dataset files under dataset/LuoDTI/data/.
2. Run data_process_5.py to generate fold-specific preprocessed tensors.
3. Run train_5.py to train and evaluate GCNDIADTI.

Example commands:

python data_process_5.py
python train_5.py

## Methodology

The preprocessing and modeling workflow is as follows:

1. Load the preprocessed MIDTI-format DTINet/Luo data files.
2. Use known DTIs as positive samples.
3. Randomly sample an equal number of unknown drug-target pairs as negative samples.
4. Perform five repeated 5-fold cross-validation experiments.
5. For each fold, construct training and test samples.
6. Build drug and target similarity views using static similarity information and fold-specific dynamic similarity information.
7. Fuse multiple drug and target similarity views.
8. Construct normalized adjacency matrices for graph representation learning.
9. Train GCNDIADTI using graph convolutional representation learning and the deep interactive attention module.
10. Evaluate the held-out test fold and summarize the results across repeated experiments.

The main training settings are:

learning rate: 5e-5
batch size: 64
maximum epochs: 80
number of repeats: 5
number of folds: 5
optimizer: Adam
weight decay: 1e-4
label smoothing: 0.1
early stopping patience: 20
attention heads: 2
interactive attention depth: 1

## Data and Code Availability

The source code and data used in this study are available at https://github.com/Chen-NingZhang/GCNDIADTI. The repository contains the GCNDIADTI implementation, the required input data files, and the README instructions for reproducing the experiments.
## Citations

Luo Y, Zhao X, Zhou J, Yang J, Zhang Y, Kuang W, Peng J, Chen L, Zeng J. 2017. A network integration approach for drug-target interaction prediction and computational drug repositioning from heterogeneous information. Nature Communications 8:573. DOI: 10.1038/s41467-017-00680-8.

Song W, Xu L, Han C, Tian Z. 2024. Drug-target interaction predictions with multi-view similarity network fusion strategy and deep interactive attention mechanism. Bioinformatics 40:btae346. DOI: 10.1093/bioinformatics/btae346.

Yamanishi Y, Araki M, Gutteridge A, Honda W, Kanehisa M. 2008. Prediction of drug-target interaction networks from the integration of chemical and genomic spaces. Bioinformatics 24:i232-i240. DOI: 10.1093/bioinformatics/btn162.

## License and Contribution Guidelines

No explicit license file was provided with the code package. If the code is deposited in a public repository, the authors should add an appropriate license statement. For contributions, contact the corresponding author.

