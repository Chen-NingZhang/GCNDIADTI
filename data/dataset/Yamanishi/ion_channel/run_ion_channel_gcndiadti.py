"""Run GCNDIADTI on the Yamanishi ion channel subset.

Place this script in data/dataset/Yamanishi/ion_channel and run:

python run_ion_channel_gcndiadti.py

The script combines the parameter setup, preprocessing, and training steps for
the ion channel subset. It imports the main GCNDIADTI model files from the project
code directory and injects ion-channel-specific dimensions before importing them.
"""

import argparse
import os
import random
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.utils.data import DataLoader, Dataset


ADJACENCY_FILE = "Adjacency matrix of the gold standard drug-target interaction data.txt"
DRUG_SIM_FILE = "Compound structure similarity matrix.txt"
PROTEIN_SIM_FILE = "Protein sequence similarity matrix.txt"


def parse_cli():
    script_dir = Path(__file__).resolve().parent
    default_device = "cuda:0" if torch.cuda.is_available() else "cpu"

    parser = argparse.ArgumentParser(description="GCNDIADTI Yamanishi ion channel runner")
    parser.add_argument("--data-dir", default=str(script_dir), type=str)
    parser.add_argument("--model-dir", default=None, type=str,
                        help="Directory containing model_all.py, Model517.py, Transformer.py, and early_stopping.py.")
    parser.add_argument("--device", default=default_device, type=str)
    parser.add_argument("--seed", default=1206, type=int)
    parser.add_argument("--folds", default=5, type=int)
    parser.add_argument("--num-repeats", default=5, type=int)
    parser.add_argument("--epochs", default=80, type=int)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--learn-rate", default=5e-5, type=float)
    parser.add_argument("--drug-num", default=210, type=int)
    parser.add_argument("--protein-num", default=204, type=int)
    parser.add_argument("--rebuild", action="store_true",
                        help="Regenerate preprocessed repeat files even if they already exist.")
    parser.add_argument("--preprocess-only", action="store_true",
                        help="Only generate preprocessed repeat files and skip training.")
    parser.add_argument("--skip-preprocess", action="store_true",
                        help="Skip preprocessing and train from existing preprocessed repeat files.")
    return parser.parse_args()


def find_model_dir(script_dir, explicit_model_dir):
    if explicit_model_dir:
        model_dir = Path(explicit_model_dir).resolve()
        if (model_dir / "model_all.py").exists():
            return model_dir
        raise FileNotFoundError("model_all.py was not found in --model-dir: %s" % model_dir)

    for parent in [script_dir] + list(script_dir.parents):
        if (parent / "model_all.py").exists() and (parent / "Model517.py").exists():
            return parent

    fallback = Path(r"D:\科研\GCNDIADTI(第三次）\code")
    if (fallback / "model_all.py").exists():
        return fallback

    raise FileNotFoundError("Could not locate the main GCNDIADTI code directory.")


def make_model_args(cli_args):
    return argparse.Namespace(
        latdim=cli_args.drug_num + cli_args.protein_num,
        drug_num=cli_args.drug_num,
        protein_num=cli_args.protein_num,
        gnn_layer=3,
        droprate=0.5,
        node_type_dim=32,
        hyperNum=64,
        attention_heads=2,
        head_dim=64,
        embed_dropout=0.3,
        attention_dropout=0.3,
        depth_interact_attention=1,
        mlp_dim=512,
        label_smoothing=0.1,
        device=cli_args.device,
    )


def install_parameter_module(model_args):
    module = types.ModuleType("parameters")
    module.args = model_args
    sys.modules["parameters"] = module


def import_main_code(model_dir):
    model_dir = str(model_dir)
    if model_dir in sys.path:
        sys.path.remove(model_dir)
    sys.path.insert(0, model_dir)

    for name in ["model_all", "Model517", "Transformer", "early_stopping"]:
        sys.modules.pop(name, None)

    from early_stopping import EarlyStopping
    from model_all import final_model
    return final_model, EarlyStopping


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tensor_shuffle(ts):
    return ts[torch.randperm(ts.shape[0], device=ts.device)]


def load_adjacency_matrix(path, drug_num, protein_num, device):
    df = pd.read_csv(path, sep=r"\s+", index_col=0)
    arr = df.values.astype(np.float32)

    if arr.shape == (protein_num, drug_num):
        drug_ids = list(df.columns)
        protein_ids = list(df.index)
        arr = arr.T
    elif arr.shape == (drug_num, protein_num):
        drug_ids = list(df.index)
        protein_ids = list(df.columns)
    else:
        raise ValueError(
            "Unexpected adjacency matrix shape %s. Expected (%d, %d) or (%d, %d)." %
            (arr.shape, protein_num, drug_num, drug_num, protein_num)
        )

    return torch.from_numpy(arr).float().to(device), drug_ids, protein_ids


def load_square_similarity(path, labels, expected_size, device, add_noise=True,
                           noise_std=0.01, drop_edge_prob=0.05):
    if not path.exists():
        print("[warn] Similarity file not found and will be skipped: %s" % path.name)
        return None

    df = pd.read_csv(path, sep=r"\s+", index_col=0)
    if labels and set(labels).issubset(set(df.index)) and set(labels).issubset(set(df.columns)):
        df = df.loc[labels, labels]

    if df.shape != (expected_size, expected_size):
        raise ValueError(
            "Unexpected shape for %s: %s. Expected (%d, %d)." %
            (path.name, df.shape, expected_size, expected_size)
        )

    sim = torch.from_numpy(df.values.astype(np.float32)).float().to(device)
    sim = torch.clamp((sim + sim.T) / 2, 0, 1)

    if add_noise:
        noise = torch.normal(mean=0.0, std=noise_std, size=sim.shape, device=device)
        sim = torch.clamp(sim + noise, 0, 1)
    if drop_edge_prob > 0:
        mask = torch.rand(sim.shape, device=device) > drop_edge_prob
        sim = sim * mask
    return sim


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, adj, inputs):
        support = torch.matmul(inputs, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            output = output + self.bias
        return output


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
    if k <= 0:
        return sim
    _, topk_idx = torch.topk(sim, k=k, dim=1)
    mask = torch.zeros_like(sim)
    mask.scatter_(1, topk_idx, 1.0)
    sparse = sim * mask
    return torch.maximum(sparse, sparse.T)


def preprocess_adj(adj, device):
    adj = adj + torch.eye(adj.shape[0], device=device)
    rowsum = torch.sum(adj, dim=1)
    d_inv_sqrt = torch.pow(rowsum, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    adj_normalized = d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt
    if torch.any(torch.isnan(adj_normalized)) or torch.any(torch.isinf(adj_normalized)):
        raise ValueError("Invalid values in normalized adjacency matrix")
    return adj_normalized


def preprocess_bipartite_adj(adj, device):
    row_sum = torch.sum(adj, dim=1) + 1e-8
    col_sum = torch.sum(adj, dim=0) + 1e-8
    row_inv_sqrt = torch.pow(row_sum, -0.5)
    col_inv_sqrt = torch.pow(col_sum, -0.5)
    adj_normalized = adj * row_inv_sqrt.unsqueeze(1) * col_inv_sqrt.unsqueeze(0)
    if torch.any(torch.isnan(adj_normalized)) or torch.any(torch.isinf(adj_normalized)):
        raise ValueError("Invalid values in normalized bipartite adjacency matrix")
    return adj_normalized


def fuse_similarity_networks(sims, num_nodes, view_count, f=512, out_channels=None, device="cpu"):
    if out_channels is None:
        out_channels = num_nodes
    if not sims:
        raise ValueError("At least one similarity view is required.")

    gcn1 = GraphConvolution(f, f).to(device)
    gcn2 = GraphConvolution(f, f).to(device)
    gcn3 = GraphConvolution(f, f).to(device)
    global_avg_pool = nn.AdaptiveAvgPool2d(1).to(device)
    fc1 = nn.Linear(view_count, 5 * view_count).to(device)
    fc2 = nn.Linear(5 * view_count, view_count).to(device)
    cnn = nn.Conv1d(view_count * f, out_channels, kernel_size=1, stride=1, bias=True).to(device)
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

    xm = torch.cat(embeddings, dim=0).view(1, view_count, num_nodes, f)
    channel_attention = global_avg_pool(xm).view(1, view_count)
    channel_attention = F.relu(fc1(channel_attention))
    channel_attention = sigmoid(fc2(channel_attention)).view(1, view_count, 1, 1)

    xm_attn = xm * channel_attention.expand_as(xm)
    xm_attn = xm_attn.view(1, view_count * f, num_nodes)
    x_fe = cnn(xm_attn).view(out_channels, num_nodes).T

    fused_sim = torch.matmul(x_fe, x_fe.T)
    fused_sim = (fused_sim + fused_sim.T) / 2
    return torch.clamp(fused_sim, 0, 1)


def build_dynamic_drug_views(dp_train, k):
    return [knn_sparsify(jaccard_from_profile(dp_train), k=k),
            knn_sparsify(gip_kernel(dp_train), k=k)]


def build_dynamic_protein_views(dp_train, k):
    return [knn_sparsify(jaccard_from_profile(dp_train.T), k=k),
            knn_sparsify(gip_kernel(dp_train.T), k=k)]


def preprocess_repeats(cli_args, model_args, data_dir, device):
    adjacency_path = data_dir / ADJACENCY_FILE
    if not adjacency_path.exists():
        raise FileNotFoundError("Required adjacency file not found: %s" % adjacency_path)

    dp, drug_ids, protein_ids = load_adjacency_matrix(
        adjacency_path, model_args.drug_num, model_args.protein_num, device
    )
    print("Loaded ion channel adjacency matrix as drug x protein: %s" % (tuple(dp.shape),))

    static_drug_sims = []
    drug_sim = load_square_similarity(
        data_dir / DRUG_SIM_FILE, drug_ids, model_args.drug_num, device,
    )
    if drug_sim is not None:
        static_drug_sims.append(knn_sparsify(drug_sim, k=30))

    static_protein_sims = []
    protein_sim = load_square_similarity(
        data_dir / PROTEIN_SIM_FILE, protein_ids, model_args.protein_num, device,
    )
    if protein_sim is not None:
        static_protein_sims.append(knn_sparsify(protein_sim, k=50))

    print("Static drug similarity views: %d" % len(static_drug_sims))
    print("Static protein similarity views: %d" % len(static_protein_sims))

    pos_index = dp.nonzero()
    neg_all = (dp == 0).nonzero()
    print("Positive DTIs: %d" % len(pos_index))

    for repeat in range(cli_args.num_repeats):
        save_filename = data_dir / ("embed_index_adj_protein_drug_1to1_strict_repeat_%d.pth" % repeat)
        if save_filename.exists() and not cli_args.rebuild:
            print("[skip] existing preprocessed file: %s" % save_filename.name)
            continue

        current_seed = cli_args.seed + repeat
        set_seed(current_seed)
        print("[Repeat %d/%d] preprocessing" % (repeat + 1, cli_args.num_repeats))

        neg_index = tensor_shuffle(neg_all)[: len(pos_index)]
        pos_shuffled = pos_index[torch.randperm(len(pos_index), device=device)]
        neg_shuffled = neg_index[torch.randperm(len(neg_index), device=device)]

        pos_fold_indices = torch.arange(len(pos_shuffled), device=device) % cli_args.folds
        neg_fold_indices = torch.arange(len(neg_shuffled), device=device) % cli_args.folds

        train_index_list = []
        test_index_list = []
        double_dm_masked = []
        embedding_list = []
        d_sim_list = []
        p_sim_list = []

        for fold in range(cli_args.folds):
            print("  Fold %d/%d: building features" % (fold + 1, cli_args.folds))
            test_pos = pos_shuffled[pos_fold_indices == fold]
            train_pos = pos_shuffled[pos_fold_indices != fold]
            test_neg = neg_shuffled[neg_fold_indices == fold]
            train_neg = neg_shuffled[neg_fold_indices != fold]

            test_pos_label = torch.cat([test_pos, torch.ones(len(test_pos), 1, device=device)], dim=1)
            train_pos_label = torch.cat([train_pos, torch.ones(len(train_pos), 1, device=device)], dim=1)
            test_neg_label = torch.cat([test_neg, torch.zeros(len(test_neg), 1, device=device)], dim=1)
            train_neg_label = torch.cat([train_neg, torch.zeros(len(train_neg), 1, device=device)], dim=1)

            train_data = tensor_shuffle(torch.cat([train_pos_label, train_neg_label], dim=0)).long()
            test_data = tensor_shuffle(torch.cat([test_pos_label, test_neg_label], dim=0)).long()
            train_index_list.append(train_data)
            test_index_list.append(test_data)

            dp_train = torch.zeros_like(dp, device=device)
            rows = train_pos[:, 0].long()
            cols = train_pos[:, 1].long()
            dp_train[rows, cols] = 1.0

            drug_sims = list(static_drug_sims) + build_dynamic_drug_views(dp_train, k=30)
            protein_sims = list(static_protein_sims) + build_dynamic_protein_views(dp_train, k=50)

            fold_drug_sim = fuse_similarity_networks(
                drug_sims, num_nodes=model_args.drug_num, view_count=len(drug_sims),
                f=512, out_channels=model_args.drug_num, device=device,
            )
            fold_protein_sim = fuse_similarity_networks(
                protein_sims, num_nodes=model_args.protein_num, view_count=len(protein_sims),
                f=512, out_channels=model_args.protein_num, device=device,
            )

            d_sim_list.append(fold_drug_sim.detach())
            p_sim_list.append(fold_protein_sim.detach())

            dp_norm = preprocess_bipartite_adj(dp_train, device=device)
            o1 = torch.zeros(model_args.drug_num, model_args.drug_num, device=device)
            o2 = torch.zeros(model_args.protein_num, model_args.protein_num, device=device)
            double_dm = torch.cat([torch.cat([o1, dp_norm], dim=1),
                                   torch.cat([dp_norm.T, o2], dim=1)], dim=0)
            double_dm_masked.append(double_dm.detach())

            embed = torch.cat([torch.cat([fold_drug_sim, dp_train], dim=1),
                               torch.cat([dp_train.T, fold_protein_sim], dim=1)], dim=0)
            embedding_list.append(embed.detach())

        torch.save(
            [embedding_list, train_index_list, test_index_list,
             double_dm_masked, dp, d_sim_list, p_sim_list],
            str(save_filename),
        )
        print("  [done] saved: %s" % save_filename.name)


class PairDataset(Dataset):
    def __init__(self, tri):
        self.tri = tri

    def __getitem__(self, idx):
        x1, x2, label = self.tri[idx, :]
        return x1, x2, label

    def __len__(self):
        return self.tri.shape[0]


def test_model(model, test_set, fold, repeat, embeds, adj, d_adj, p_adj, device):
    predall, yall = torch.tensor([]), torch.tensor([])
    model.eval()
    best_model_path = Path("best_parameter") / ("repeat_%d" % repeat) / ("fold_%d" % fold) / "best_network_auc.pth"

    if best_model_path.exists():
        model.load_state_dict(torch.load(str(best_model_path), map_location=device))

    with torch.no_grad():
        for x1, x2, y in test_set:
            x1 = x1.long().to(device)
            x2 = x2.long().to(device)
            y = y.long().to(device)
            pred = model(x1, x2, embeds, adj, d_adj, p_adj)
            predall = torch.cat([predall, torch.as_tensor(pred, device="cpu")], dim=0)
            yall = torch.cat([yall, torch.as_tensor(y, device="cpu")])

    pred_probs = torch.softmax(predall, dim=1)[:, 1]
    fpr, tpr, thresholds = roc_curve(yall.numpy(), pred_probs.numpy())
    optimal_idx = np.argmax(tpr - fpr)
    optimal_th = thresholds[optimal_idx]

    pred_label = (pred_probs >= optimal_th).float()
    test_acc = (pred_label == yall).sum().item() / yall.size(0)
    roc_auc = roc_auc_score(yall.numpy(), pred_probs.numpy())
    pr_aupr = average_precision_score(yall.numpy(), pred_probs.numpy())

    print(">>> Result: Repeat %d | Fold %d -> AUC: %.6f, AUPR: %.6f, ACC: %.6f" %
          (repeat + 1, fold + 1, roc_auc, pr_aupr, test_acc))

    result_dir = Path("result") / ("repeat_%d" % repeat)
    result_dir.mkdir(parents=True, exist_ok=True)
    torch.save((predall, yall), str(result_dir / ("fold_%d" % fold)))

    return test_acc, roc_auc, pr_aupr


def train_one_fold(model, train_set, test_set, embed, epoch, learn_rate, fold, repeat,
                   adj_tri, d_adj, p_adj, device, label_smoothing, EarlyStopping):
    transformer_params = list(model.transformer.parameters())
    other_params = [p for n, p in model.named_parameters() if "transformer" not in n]
    optimizer = torch.optim.Adam([
        {"params": other_params, "lr": learn_rate},
        {"params": transformer_params, "lr": learn_rate * 0.1},
    ], weight_decay=1e-4)

    cost = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    embeds = embed.float().to(device)
    adj_tri = adj_tri.float().to(device)
    d_adj = d_adj.float().to(device)
    p_adj = p_adj.float().to(device)

    save_dir = Path("best_parameter") / ("repeat_%d" % repeat) / ("fold_%d" % fold)
    early_stopping = EarlyStopping(patience=20, verbose=False, save_path=str(save_dir), delta=0.001)

    warmup_epochs = 2
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, epoch - warmup_epochs))
    warmup_scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda e: e / warmup_epochs if e < warmup_epochs else 1,
    )

    best_val_auc = 0.0
    best_model_path = save_dir / "best_network_auc.pth"
    grace_epochs = 10

    for i in range(epoch):
        model.train()
        train_loss = 0.0
        for x1, x2, y in train_set:
            x1 = x1.long().to(device)
            x2 = x2.long().to(device)
            y = y.long().to(device)
            out = model(x1, x2, embeds, adj_tri, d_adj, p_adj)
            loss = cost(out, y)
            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        train_loss /= max(1, len(train_set))

        model.eval()
        val_pred, val_y = [], []
        with torch.no_grad():
            for x1, x2, y in test_set:
                x1 = x1.long().to(device)
                x2 = x2.long().to(device)
                y = y.long().to(device)
                out = model(x1, x2, embeds, adj_tri, d_adj, p_adj)
                val_pred.append(torch.softmax(out, dim=1)[:, 1])
                val_y.append(y)

        val_pred = torch.cat(val_pred).cpu().numpy()
        val_y = torch.cat(val_y).cpu().numpy()
        val_auc = roc_auc_score(val_y, val_pred)

        if (i + 1) % 10 == 0:
            print("Repeat: %d | Fold: %d | Epoch: %d | Loss: %.5f | Val AUC: %.5f" %
                  (repeat + 1, fold + 1, i + 1, train_loss, val_auc))

        if i >= grace_epochs:
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                save_dir.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), str(best_model_path))
            early_stopping(-val_auc, model)
            if early_stopping.early_stop:
                print("  [Repeat %d-Fold %d] Early stopping at epoch %d" %
                      (repeat + 1, fold + 1, i + 1))
                return test_model(model, test_set, fold, repeat, embeds, adj_tri, d_adj, p_adj, device)
        elif i == grace_epochs - 1:
            best_val_auc = val_auc
            save_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), str(best_model_path))

        if i + 1 == epoch:
            return test_model(model, test_set, fold, repeat, embeds, adj_tri, d_adj, p_adj, device)

        if i < warmup_epochs:
            warmup_scheduler.step()
        else:
            scheduler.step()


def run_training(cli_args, model_args, data_dir, device, final_model, EarlyStopping):
    grand_accs, grand_aucs, grand_auprs = [], [], []
    print("========== Starting %d repetitions of %d-fold cross-validation ==========" %
          (cli_args.num_repeats, cli_args.folds))

    for repeat in range(cli_args.num_repeats):
        print("\n######################################################")
        print("Start Repeat Experiment %d / %d" % (repeat + 1, cli_args.num_repeats))
        print("######################################################")

        data_filename = data_dir / ("embed_index_adj_protein_drug_1to1_strict_repeat_%d.pth" % repeat)
        if not data_filename.exists():
            raise FileNotFoundError("Missing preprocessed file: %s" % data_filename)

        embed, train_index, test_index, masked_dp, dp, d_list, p_list = torch.load(
            str(data_filename), map_location=device
        )
        repeat_accs, repeat_aucs, repeat_auprs = [], [], []

        for fold in range(cli_args.folds):
            set_seed(cli_args.seed + repeat * 100 + fold)
            net = final_model().to(device)
            train_set = DataLoader(PairDataset(train_index[fold]), cli_args.batch_size, shuffle=True)
            test_set = DataLoader(PairDataset(test_index[fold]), cli_args.batch_size, shuffle=False)

            test_acc, roc_auc, pr_aupr = train_one_fold(
                net, train_set, test_set, embed[fold], cli_args.epochs, cli_args.learn_rate,
                fold, repeat, masked_dp[fold], d_list[fold], p_list[fold], device,
                model_args.label_smoothing, EarlyStopping,
            )
            repeat_accs.append(test_acc)
            repeat_aucs.append(roc_auc)
            repeat_auprs.append(pr_aupr)

        avg_acc = np.mean(repeat_accs)
        avg_auc = np.mean(repeat_aucs)
        avg_aupr = np.mean(repeat_auprs)
        grand_accs.append(avg_acc)
        grand_aucs.append(avg_auc)
        grand_auprs.append(avg_aupr)

        print("\n[Summary Repeat %d] Avg ACC: %.6f | Avg AUC: %.6f | Avg AUPR: %.6f" %
              (repeat + 1, avg_acc, avg_auc, avg_aupr))

    print("\n\n==========================================================")
    print("Final Results over %d runs of %d-fold CV:" % (cli_args.num_repeats, cli_args.folds))
    print("==========================================================")
    print("ACC : %.6f +/- %.6f" % (np.mean(grand_accs), np.std(grand_accs)))
    print("AUC : %.6f +/- %.6f" % (np.mean(grand_aucs), np.std(grand_aucs)))
    print("AUPR: %.6f +/- %.6f" % (np.mean(grand_auprs), np.std(grand_auprs)))
    print("==========================================================")


def main():
    cli_args = parse_cli()
    data_dir = Path(cli_args.data_dir).resolve()
    script_dir = Path(__file__).resolve().parent
    model_dir = find_model_dir(script_dir, cli_args.model_dir)
    model_args = make_model_args(cli_args)
    device = torch.device(model_args.device)

    os.chdir(str(data_dir))
    set_seed(cli_args.seed)
    install_parameter_module(model_args)
    final_model, EarlyStopping = import_main_code(model_dir)

    print("Data directory: %s" % data_dir)
    print("Main model directory: %s" % model_dir)
    print("Device: %s" % device)
    print("Ion channel dimensions: drug_num=%d, protein_num=%d, latdim=%d" %
          (model_args.drug_num, model_args.protein_num, model_args.latdim))
    print("Main hyperparameters: lr=%g, heads=%d, depth=%d, mlp_dim=%d" %
          (cli_args.learn_rate, model_args.attention_heads,
           model_args.depth_interact_attention, model_args.mlp_dim))

    if not cli_args.skip_preprocess:
        preprocess_repeats(cli_args, model_args, data_dir, device)

    if cli_args.preprocess_only:
        print("Preprocessing finished. Training skipped because --preprocess-only was set.")
        return

    run_training(cli_args, model_args, data_dir, device, final_model, EarlyStopping)


if __name__ == "__main__":
    main()