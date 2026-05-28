# GCNDIADTI

## Title

GCNDIADTI: Graph Convolutional Networks and Deep Interactive Attention Module for Drug-Target Interaction Prediction

## Description

This README describes the code and data files used to reproduce the GCNDIADTI experiments. GCNDIADTI is a drug-target interaction prediction model that integrates multi-source drug similarity information, multi-source target similarity information, known drug-target associations, graph convolutional representation learning, and a deep interactive attention module.

## Dataset Information

The main benchmark dataset used in this study is the DTINet/Luo dataset. The original DTINet/Luo dataset was introduced by Luo et al. (2017), and the DTINet repository is available at https://github.com/luoyunan/DTINet. The Luo benchmark used in this study contains 1,923 known drug-target interactions between 708 drugs and 1,512 targets.

Original DTINet/Luo data source:

Luo Y, Zhao X, Zhou J, Yang J, Zhang Y, Kuang W, Peng J, Chen L, Zeng J. 2017. A network integration approach for drug-target interaction prediction and computational drug repositioning from heterogeneous information. Nature Communications 8:573. DOI: 10.1038/s41467-017-00680-8.

DTINet repository: https://github.com/luoyunan/DTINet

The Yamanishi et al. (2008) benchmark dataset was used for generalization experiments. The enzyme and ion channel subsets were used in this study.

Yamanishi data source:

Yamanishi Y, Araki M, Gutteridge A, Honda W, Kanehisa M. 2008. Prediction of drug-target interaction networks from the integration of chemical and genomic spaces. Bioinformatics 24:i232-i240. DOI: 10.1093/bioinformatics/btn162.

The input files required by the Luo experiment are placed under data/dataset/LuoDTI/data/:

protein_drug_interaction.txt
drug_drug_interaction.csv
drug_disease_association.csv
drug_side_effect_association.csv
drug_chemical_structure.csv
protein_protein_interaction.csv
protein_disease_association.csv
protein_genome_sequence.csv

The input files required by the Yamanishi enzyme experiment are placed under data/dataset/Yamanishi/enzyme/:

Adjacency matrix of the gold standard drug-target interaction data.txt
Binary relation list of the gold standard drug-target interaction data.txt
Compound structure similarity matrix.txt
Protein sequence similarity matrix.txt

The input files required by the Yamanishi ion channel experiment are placed under data/dataset/Yamanishi/ion_channel/:

Adjacency matrix of the gold standard drug-target interaction data.txt
Binary relation list of the gold standard drug-target interaction data.txt
Compound structure similarity matrix.txt
Protein sequence similarity matrix.txt

## Code Information

The main GCNDIADTI implementation is provided in the project code directory:

data_process_5.py: prepares the fold-specific input tensors for the Luo benchmark.
train_5.py: trains and evaluates GCNDIADTI on the Luo benchmark and reports ACC, ROC-AUC, and AUPR.
model_all.py: defines the complete GCNDIADTI model.
Model517.py: defines graph convolution, feature integration, and classifier modules.
Transformer.py: defines the deep interactive attention module.
early_stopping.py: implements early stopping and checkpoint saving.
parameters.py: stores default model parameters and device settings.

The Yamanishi generalization experiments are provided as dataset-specific single-runner scripts:

data/dataset/Yamanishi/enzyme/run_enzyme_gcndiadti.py: combines parameter setup, preprocessing, and training for the Yamanishi enzyme subset.
data/dataset/Yamanishi/ion_channel/run_ion_channel_gcndiadti.py: combines parameter setup, preprocessing, and training for the Yamanishi ion channel subset.

Running the scripts generates the following outputs:

embed_index_adj_protein_drug_1to1_strict_repeat_0.pth to embed_index_adj_protein_drug_1to1_strict_repeat_4.pth: preprocessed fold-specific tensors.
best_parameter/: saved model checkpoints.
result/: saved prediction outputs.

These generated outputs are not required before running the code and can be regenerated from the provided data and scripts.

## Requirements

The code requires Python and the following Python packages:

Python 3.7 or later
torch==1.10.1
numpy==1.21.6
pandas==1.1.5
scikit-learn==1.0.2
einops==0.6.1

## Usage Instructions

For the Luo benchmark, run the scripts from the project code directory in the following order:

1. Prepare the dataset files under data/dataset/LuoDTI/data/.
2. Run data_process_5.py to generate fold-specific preprocessed tensors.
3. Run train_5.py to train and evaluate GCNDIADTI.

Example commands:

python data_process_5.py
python train_5.py

For the Yamanishi enzyme experiment, run the following command from data/dataset/Yamanishi/enzyme/:

python run_enzyme_gcndiadti.py

For the Yamanishi ion channel experiment, run the following command from data/dataset/Yamanishi/ion_channel/:

python run_ion_channel_gcndiadti.py

## Methodology

The preprocessing and modeling workflow is as follows:

1. Load the benchmark data files.
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

## Citations

Luo Y, Zhao X, Zhou J, Yang J, Zhang Y, Kuang W, Peng J, Chen L, Zeng J. 2017. A network integration approach for drug-target interaction prediction and computational drug repositioning from heterogeneous information. Nature Communications 8:573. DOI: 10.1038/s41467-017-00680-8.

Yamanishi Y, Araki M, Gutteridge A, Honda W, Kanehisa M. 2008. Prediction of drug-target interaction networks from the integration of chemical and genomic spaces. Bioinformatics 24:i232-i240. DOI: 10.1093/bioinformatics/btn162.

##License and Contribution Guidelines
The source code is released under the MIT License. The third-party Luo/DTINet and Yamanishi benchmark datasets are provided for reproducibility and should be cited using their original publications. For contributions or questions, please contact the corresponding author.