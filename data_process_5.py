import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F

from parameters import args


base_seed = 1206
torch.manual_seed(base_seed)


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, adj, input):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            output = output + self.bias
        return output


def load_static_similarity(file, add_noise=True, noise_std=0.01,
                           drop_edge_prob=0.05, device=args.device):
    sim = torch.from_numpy(
        pd.read_csv(f"dataset/LuoDTI/data/{file}", header=None).values
    ).float().to(device)
    if add_noise:
        noise = torch.normal(mean=0.0, std=noise_std, size=sim.shape, device=device)
        sim = torch.clamp(sim + noise, 0, 1)
    if drop_edge_prob > 0:
        mask = torch.rand(sim.shape, device=device) > drop_edge_prob
        sim = sim * mask
    return sim


def gip_kernel(profile, gamma_factor=1.0):
    profile = profile.float()
    norms_sq = (profile * profile).sum(dim=1)
    avg = norms_sq.mean().clamp(min=1e-8)
    gamma = gamma_factor / avg
    dist_sq = norms_sq.unsqueeze(0) + norms_sq.unsqueeze(1) - 2.0 * (profile @ profile.T)
    dist_sq = torch.clamp(dist_sq, min=0.0)
    return torch.exp(-gamma * dist_sq)


def jaccard_from_profile(profile, eps=1e-8):
    profile = profile.float()
    inter = profile @ profile.T
    diag = inter.diag()
    union = diag.unsqueeze(0) + diag.unsqueeze(1) - inter + eps
    sim = inter / union
    return torch.clamp(sim, 0, 1)


def knn_sparsify(sim, k=30):
    n = sim.shape[0]
    k = min(k, n - 1)
    _, topk_idx = torch.topk(sim, k=k, dim=1)
    mask = torch.zeros_like(sim)
    mask.scatter_(1, topk_idx, 1.0)
    sparse = sim * mask
    sparse = torch.maximum(sparse, sparse.T)
    return sparse


def preprocess_adj(adj, device=args.device):
    adj = adj + torch.eye(adj.shape[0], device=device)
    rowsum = torch.sum(adj, dim=1)
    d_inv_sqrt = torch.pow(rowsum, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    adj_normalized = d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt
    if torch.any(torch.isnan(adj_normalized)) or torch.any(torch.isinf(adj_normalized)):
        raise ValueError("Invalid values in normalized adjacency matrix")
    return adj_normalized


def preprocess_bipartite_adj(adj, device=args.device):
    row_sum = torch.sum(adj, dim=1) + 1e-8
    col_sum = torch.sum(adj, dim=0) + 1e-8
    row_inv_sqrt = torch.pow(row_sum, -0.5)
    col_inv_sqrt = torch.pow(col_sum, -0.5)
    adj_normalized = adj * row_inv_sqrt.unsqueeze(1) * col_inv_sqrt.unsqueeze(0)
    if torch.any(torch.isnan(adj_normalized)) or torch.any(torch.isinf(adj_normalized)):
        raise ValueError("Invalid values in normalized bipartite adjacency matrix")
    return adj_normalized


def fuse_similarity_networks(sims, num_nodes, view_d, f=512,
                             out_channels=None, device=args.device):
    if out_channels is None:
        out_channels = num_nodes

    gcn1 = GraphConvolution(f, f).to(device)
    gcn2 = GraphConvolution(f, f).to(device)
    gcn3 = GraphConvolution(f, f).to(device)
    global_avg_pool = nn.AdaptiveAvgPool2d(1).to(device)
    fc1 = nn.Linear(view_d, 5 * view_d).to(device)
    fc2 = nn.Linear(5 * view_d, view_d).to(device)
    cnn = nn.Conv1d(view_d * f, out_channels, kernel_size=1, stride=1, bias=True).to(device)
    sigmoid = nn.Sigmoid().to(device)

    input_projection = nn.Linear(num_nodes, f).to(device)
    x = torch.eye(num_nodes, device=device)
    x = input_projection(x)

    embeddings = []
    for sim in sims:
        sim_n = preprocess_adj(sim, device=device)
        x1 = F.relu(gcn1(sim_n, x))
        x2 = F.relu(gcn2(sim_n, x1))
        x3 = F.relu(gcn3(sim_n, x2))
        embeddings.append(x3.unsqueeze(0))
    XM = torch.cat(embeddings, dim=0)
    XM = XM.view(1, view_d, num_nodes, f)

    x_channel_attention = global_avg_pool(XM).view(1, view_d)
    x_channel_attention = F.relu(fc1(x_channel_attention))
    x_channel_attention = sigmoid(fc2(x_channel_attention)).view(1, view_d, 1, 1)

    XM_attn = XM * x_channel_attention.expand_as(XM)
    XM_attn = XM_attn.view(1, view_d * f, num_nodes)
    x_fe = cnn(XM_attn).view(out_channels, num_nodes).t()

    fused_sim = torch.matmul(x_fe, x_fe.t())
    fused_sim = (fused_sim + fused_sim.T) / 2
    return torch.clamp(fused_sim, 0, 1)


def tensor_shuffle(ts, dim=0):
    return ts[torch.randperm(ts.shape[dim])]


drug_static_files = [
    "drug_drug_interaction.csv",
    "drug_disease_association.csv",
    "drug_side_effect_association.csv",
    "drug_chemical_structure.csv",
]
protein_static_files = [
    "protein_protein_interaction.csv",
    "protein_disease_association.csv",
    "protein_genome_sequence.csv",
]

f = 512
nfold = 5
num_repeats = 5
device = args.device

add_noise = True
noise_std = 0.01
drop_edge_prob = 0.05

knn_k_drug = 30
knn_k_protein = 50

view_d_total = len(drug_static_files) + 2
view_p_total = len(protein_static_files) + 2

drug_static_sims = [
    knn_sparsify(load_static_similarity(file, add_noise=add_noise,
                                        noise_std=noise_std,
                                        drop_edge_prob=drop_edge_prob,
                                        device=device),
                 k=knn_k_drug)
    for file in drug_static_files
]
protein_static_sims = [
    knn_sparsify(load_static_similarity(file, add_noise=add_noise,
                                        noise_std=noise_std,
                                        drop_edge_prob=drop_edge_prob,
                                        device=device),
                 k=knn_k_protein)
    for file in protein_static_files
]

DP = torch.from_numpy(
    np.loadtxt("dataset/LuoDTI/data/protein_drug_interaction.txt", delimiter=' ')
).float().to(device)


def build_dynamic_drug_views(DP_train, k=knn_k_drug):
    jacc = jaccard_from_profile(DP_train)
    gip = gip_kernel(DP_train)
    return [knn_sparsify(jacc, k=k), knn_sparsify(gip, k=k)]


def build_dynamic_protein_views(DP_train, k=knn_k_protein):
    jacc = jaccard_from_profile(DP_train.T)
    gip = gip_kernel(DP_train.T)
    return [knn_sparsify(jacc, k=k), knn_sparsify(gip, k=k)]


for repeat in range(num_repeats):
    current_seed = base_seed + repeat
    torch.manual_seed(current_seed)
    print(f"[Repeat {repeat + 1}/{num_repeats}]")

    pos_index = DP.nonzero()
    neg_all = (DP == 0).nonzero()
    neg_all = tensor_shuffle(neg_all)
    neg_index = neg_all[: len(pos_index)]

    pos_index = pos_index[torch.randperm(len(pos_index))]
    neg_index = neg_index[torch.randperm(len(neg_index))]

    pos_fold_indices = torch.arange(len(pos_index)) % nfold
    neg_fold_indices = torch.arange(len(neg_index)) % nfold

    train_index_list = []
    test_index_list = []
    double_DM_masked = []
    embedding_list = []
    D_sim_list = []
    P_sim_list = []

    for kfold in range(nfold):
        print(f'  Fold {kfold + 1}: building fold-specific features...')

        test_pos_k = pos_index[pos_fold_indices == kfold]
        train_pos_k = pos_index[pos_fold_indices != kfold]
        test_neg_k = neg_index[neg_fold_indices == kfold]
        train_neg_k = neg_index[neg_fold_indices != kfold]

        test_pos_label = torch.cat(
            [test_pos_k, torch.ones(len(test_pos_k), 1, device=device)], dim=1)
        train_pos_label = torch.cat(
            [train_pos_k, torch.ones(len(train_pos_k), 1, device=device)], dim=1)
        test_neg_label = torch.cat(
            [test_neg_k, torch.zeros(len(test_neg_k), 1, device=device)], dim=1)
        train_neg_label = torch.cat(
            [train_neg_k, torch.zeros(len(train_neg_k), 1, device=device)], dim=1)

        train_fold_data = tensor_shuffle(torch.cat([train_pos_label, train_neg_label], 0))
        test_fold_data = tensor_shuffle(torch.cat([test_pos_label, test_neg_label], 0))
        train_index_list.append(train_fold_data.long())
        test_index_list.append(test_fold_data.long())

        DP_train = torch.zeros_like(DP, device=device)
        rows = train_pos_k[:, 0].long()
        cols = train_pos_k[:, 1].long()
        DP_train[rows, cols] = 1.0

        cur_drug_sims = list(drug_static_sims) + build_dynamic_drug_views(DP_train)
        cur_protein_sims = list(protein_static_sims) + build_dynamic_protein_views(DP_train)

        fold_Drug_sim = fuse_similarity_networks(
            cur_drug_sims, num_nodes=args.drug_num,
            view_d=view_d_total, f=f,
            out_channels=args.drug_num, device=device,
        )
        fold_Protein_sim = fuse_similarity_networks(
            cur_protein_sims, num_nodes=args.protein_num,
            view_d=view_p_total, f=f,
            out_channels=args.protein_num, device=device,
        )

        D_sim_list.append(fold_Drug_sim.detach())
        P_sim_list.append(fold_Protein_sim.detach())

        DP_i_norm = preprocess_bipartite_adj(DP_train, device=device)
        O1 = torch.zeros(args.drug_num, args.drug_num, device=device)
        O2 = torch.zeros(args.protein_num, args.protein_num, device=device)
        double_DM = torch.cat([torch.cat([O1, DP_i_norm], 1),
                               torch.cat([DP_i_norm.T, O2], 1)], 0)
        double_DM_masked.append(double_DM)

        embed = torch.cat([torch.cat([fold_Drug_sim, DP_train], 1),
                           torch.cat([DP_train.T, fold_Protein_sim], 1)], 0)
        embedding_list.append(embed.detach())

    save_filename = f'embed_index_adj_protein_drug_1to1_strict_repeat_{repeat}.pth'
    torch.save(
        [embedding_list, train_index_list, test_index_list,
         double_DM_masked, DP, D_sim_list, P_sim_list],
        save_filename,
    )
    print(f"  [done] saved: {save_filename}")
