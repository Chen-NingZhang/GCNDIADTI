import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

from early_stopping import EarlyStopping
from model_all import final_model
from parameters import args


device = torch.device(args.device)


def train(model, train_set, test_set, embed, epoch, learn_rate, cross, repeat,
          adj_tri, D_adj, P_adj):
    transformer_params = list(model.transformer.parameters())
    other_params = [p for n, p in model.named_parameters() if "transformer" not in n]

    optimizer = torch.optim.Adam([
        {'params': other_params, 'lr': learn_rate},
        {'params': transformer_params, 'lr': learn_rate * 0.1},
    ], weight_decay=1e-4)

    cost = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    embeds = embed.float().to(device)
    adj_tri = adj_tri.float().to(device)
    D_adj = D_adj.float().to(device)
    P_adj = P_adj.float().to(device)

    save_dir = f'best_parameter/repeat_{repeat}/fold_{cross}'
    early_stopping = EarlyStopping(patience=20, verbose=False, save_path=save_dir, delta=0.001)

    warmup_epochs = 2
    scheduler = CosineAnnealingLR(optimizer, T_max=epoch - warmup_epochs)
    warmup_scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda e: e / warmup_epochs if e < warmup_epochs else 1,
    )

    best_val_auc = 0
    best_model_path = os.path.join(save_dir, 'best_network_auc.pth')
    grace_epochs = 10

    for i in range(epoch):
        model.train()
        train_loss = 0
        for x1, x2, y in train_set:
            x1, x2, y = x1.long().to(device), x2.long().to(device), y.long().to(device)
            out = model(x1, x2, embeds, adj_tri, D_adj, P_adj)
            loss = cost(out, y)
            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        train_loss /= len(train_set)

        model.eval()
        val_loss = 0
        val_pred, val_y = [], []
        with torch.no_grad():
            for x1, x2, y in test_set:
                x1, x2, y = x1.long().to(device), x2.long().to(device), y.long().to(device)
                out = model(x1, x2, embeds, adj_tri, D_adj, P_adj)
                val_loss += cost(out, y).item()
                val_pred.append(torch.softmax(out, dim=1)[:, 1])
                val_y.append(y)

        val_loss /= len(test_set)
        val_pred = torch.cat(val_pred).cpu().numpy()
        val_y = torch.cat(val_y).cpu().numpy()

        val_auc = roc_auc_score(val_y, val_pred)
        val_aupr = average_precision_score(val_y, val_pred)

        if (i + 1) % 10 == 0:
            print("Repeat: %d | Fold: %d | Epoch: %d | Loss: %.5f | Val AUC: %.5f"
                  % (repeat + 1, cross + 1, i + 1, train_loss, val_auc))

        if i >= grace_epochs:
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                os.makedirs(save_dir, exist_ok=True)
                torch.save(model.state_dict(), best_model_path)
            early_stopping(-val_auc, model)
            if early_stopping.early_stop:
                print(f'  [Repeat {repeat + 1}-Fold {cross + 1}] '
                      f'Early stopping at epoch {i + 1}')
                return test(model, test_set, cross, repeat, embeds, adj_tri, D_adj, P_adj)
        elif i == grace_epochs - 1:
            best_val_auc = val_auc
            os.makedirs(save_dir, exist_ok=True)
            torch.save(model.state_dict(), best_model_path)

        if i + 1 == epoch:
            return test(model, test_set, cross, repeat, embeds, adj_tri, D_adj, P_adj)

        if i < warmup_epochs:
            warmup_scheduler.step()
        else:
            scheduler.step()


def test(model, test_set, cross, repeat, embeds, adj, D_adj, P_adj):
    predall, yall = torch.tensor([]), torch.tensor([])
    model.eval()
    best_model_path = f'best_parameter/repeat_{repeat}/fold_{cross}/best_network_auc.pth'

    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))

    with torch.no_grad():
        for x1, x2, y in test_set:
            x1, x2, y = x1.long().to(device), x2.long().to(device), y.long().to(device)
            pred = model(x1, x2, embeds, adj, D_adj, P_adj)
            predall = torch.cat([predall, torch.as_tensor(pred, device='cpu')], dim=0)
            yall = torch.cat([yall, torch.as_tensor(y, device='cpu')])

    pred_probs = torch.softmax(predall, dim=1)[:, 1]

    fpr, tpr, thresholds = roc_curve(yall.numpy(), pred_probs.numpy())
    optimal_idx = np.argmax(tpr - fpr)
    optimal_th = thresholds[optimal_idx]

    pred_label = (pred_probs >= optimal_th).float()
    correct = (pred_label == yall).sum().item()
    total = yall.size(0)
    test_acc = correct / total

    roc_auc = roc_auc_score(yall.numpy(), pred_probs.numpy())
    pr_aupr = average_precision_score(yall.numpy(), pred_probs.numpy())

    print(f'>>> Result: Repeat {repeat + 1} | Fold {cross + 1} '
          f'-> AUC: {roc_auc:.6f}, AUPR: {pr_aupr:.6f}, ACC: {test_acc:.6f}')

    result_dir = f'result/repeat_{repeat}'
    os.makedirs(result_dir, exist_ok=True)
    torch.save((predall, yall), os.path.join(result_dir, f'fold_{cross}'))

    return test_acc, roc_auc, pr_aupr


class MyDataset(Dataset):
    def __init__(self, tri, ld):
        self.tri = tri
        self.ld = ld

    def __getitem__(self, idx):
        x, y, label = self.tri[idx, :]
        return x, y, label

    def __len__(self):
        return self.tri.shape[0]


if __name__ == "__main__":
    learn_rate = 5e-5
    epoch = 80
    batch = 64
    num_repeats = 5

    grand_accs = []
    grand_aucs = []
    grand_auprs = []

    print(f"========== Starting {num_repeats} repetitions of 5-fold cross-validation ==========")

    for repeat in range(num_repeats):
        print(f"\n######################################################")
        print(f"Start Repeat Experiment {repeat + 1} / {num_repeats}")
        print(f"######################################################")

        data_filename = f'embed_index_adj_protein_drug_1to1_strict_repeat_{repeat}.pth'
        if not os.path.exists(data_filename):
            print(f"Error: file {data_filename} does not exist.")
            break

        embed, train_index, test_index, masked_DP, DP, D_list, P_list = torch.load(data_filename)
        repeat_accs, repeat_aucs, repeat_auprs = [], [], []

        for fold in range(5):
            net = final_model().to(device)
            train_set = DataLoader(MyDataset(train_index[fold], DP), batch, shuffle=True)
            test_set = DataLoader(MyDataset(test_index[fold], DP), batch, shuffle=False)

            test_acc, roc_auc, pr_aupr = train(
                net, train_set, test_set, embed[fold], epoch, learn_rate, fold, repeat,
                masked_DP[fold], D_list[fold], P_list[fold],
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

        print(f"\n[Summary Repeat {repeat + 1}] "
              f"Avg ACC: {avg_acc:.6f} | Avg AUC: {avg_auc:.6f} | Avg AUPR: {avg_aupr:.6f}")

    print("\n\n==========================================================")
    print(f"Final Results over {num_repeats} runs of 5-fold CV:")
    print("==========================================================")
    print(f"ACC : {np.mean(grand_accs):.6f} ± {np.std(grand_accs):.6f}")
    print(f"AUC : {np.mean(grand_aucs):.6f} ± {np.std(grand_aucs):.6f}")
    print(f"AUPR: {np.mean(grand_auprs):.6f} ± {np.std(grand_auprs):.6f}")
    print("==========================================================")
