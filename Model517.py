import torch
from torch import nn
import torch.nn.functional as F
from einops import repeat

from parameters import args


init = nn.init.xavier_uniform_


class AttentionFusion(nn.Module):
    def __init__(self, hgnn_dim=512, trans_dim=512, neigh_dim=256, fused_dim=512):
        super().__init__()
        self.fused_dim = fused_dim
        self.proj_hgnn = nn.Linear(hgnn_dim, fused_dim)
        self.proj_trans = nn.Linear(trans_dim, fused_dim)
        self.proj_neigh = nn.Linear(neigh_dim, fused_dim)
        self.attention_net = nn.Sequential(
            nn.Linear(fused_dim, fused_dim // 2),
            nn.Tanh(),
            nn.Linear(fused_dim // 2, 1),
        )
        self.act = nn.LeakyReLU()

    def forward(self, x_hgnn, x_trans, x_neigh):
        h_hgnn = self.act(self.proj_hgnn(x_hgnn))
        h_trans = self.act(self.proj_trans(x_trans))
        h_neigh = self.act(self.proj_neigh(x_neigh))
        stacked = torch.stack([h_hgnn, h_trans, h_neigh], dim=3)
        scores = self.attention_net(stacked)
        weights = F.softmax(scores, dim=3)
        return torch.sum(stacked * weights, dim=3)


class HGNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.act = nn.LeakyReLU()
        self.num_layers = 3
        self.dropout = nn.Dropout(0.3)
        self.hyper_layers = nn.ParameterList()
        self.db_layers = nn.ParameterList()
        self.de_layers = nn.ParameterList()
        for _ in range(self.num_layers):
            self.hyper_layers.append(nn.Parameter(init(torch.empty(args.latdim, args.hyperNum))))
            self.db_layers.append(nn.Parameter(init(torch.empty(args.latdim, 1))))
            self.de_layers.append(nn.Parameter(init(torch.empty(args.hyperNum, 1))))
        self.Dnode_embed = nn.Parameter(torch.randn(1, args.node_type_dim))
        self.Pnode_embed = nn.Parameter(torch.randn(1, args.node_type_dim))
        self.Wd = nn.Parameter(init(torch.empty(args.drug_num, args.drug_num)))
        self.Wp = nn.Parameter(init(torch.empty(args.protein_num, args.protein_num)))
        self.linear = nn.Linear(args.latdim + args.node_type_dim, 512)

    def forward(self, x1, x2, embeds):
        Dnode_embed = repeat(self.Dnode_embed, '() e -> n e', n=args.drug_num)
        Pnode_embed = repeat(self.Pnode_embed, '() e -> n e', n=args.protein_num)
        Dnode_embed = self.Wd @ Dnode_embed
        Pnode_embed = self.Wp @ Pnode_embed
        node_type = torch.cat([Dnode_embed, Pnode_embed], dim=0)
        current_embeds = torch.cat([embeds, node_type], dim=1)
        for i in range(self.num_layers):
            residual = current_embeds
            Hyper = embeds @ self.hyper_layers[i]
            hyper_embed_update = self.act(
                (self.db_layers[i] * Hyper) @ (self.de_layers[i] * Hyper.T)
                * self.db_layers[i] @ current_embeds
            )
            current_embeds = residual + self.dropout(hyper_embed_update)
        hyper_embed = current_embeds
        x_drug = hyper_embed[x1][:, None, None, :]
        x_protein = hyper_embed[x2 + args.drug_num][:, None, None, :]
        x_drug = self.act(self.linear(x_drug))
        x_protein = self.act(self.linear(x_protein))
        return torch.cat([x_drug, x_protein], dim=2)


class MLPClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        flattened_size = 1024
        self.net = nn.Sequential(
            nn.Linear(flattened_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        return self.net(x)


class FFN_homo_drug(nn.Module):
    def __init__(self):
        super().__init__()
        self.L1 = nn.Linear(1280, 512, bias=True)
        self.L2 = nn.Linear(512, 128, bias=True)
        self.act = nn.Tanh()

    def forward(self, x):
        x = self.act(self.L1(x))
        x = self.act(self.L2(x))
        return x


class FFN_homo_prot(nn.Module):
    def __init__(self):
        super().__init__()
        self.L1 = nn.Linear(1280, 512, bias=True)
        self.L2 = nn.Linear(512, 128, bias=True)
        self.act = nn.Tanh()

    def forward(self, x):
        x = self.act(self.L1(x))
        x = self.act(self.L2(x))
        return x


class FFN_hete(nn.Module):
    def __init__(self):
        super().__init__()
        self.L1 = nn.Linear(1280, 512, bias=True)
        self.L2 = nn.Linear(512, 128, bias=True)
        self.act = nn.Tanh()

    def forward(self, x):
        x = self.act(self.L1(x))
        x = self.act(self.L2(x))
        return x


class NAPGCN(nn.Module):
    def __init__(self):
        super().__init__()
        h_dim1, h_dim2 = 768, 512
        self.wd1 = nn.Linear(args.latdim, h_dim1)
        self.wd2 = nn.Linear(h_dim1, h_dim2)
        self.wp1 = nn.Linear(args.latdim, h_dim1)
        self.wp2 = nn.Linear(h_dim1, h_dim2)
        self.dm1 = nn.Linear(args.latdim, h_dim1)
        self.dm2 = nn.Linear(h_dim1, h_dim2)

        self.norm_d1 = nn.LayerNorm(h_dim1)
        self.norm_d2 = nn.LayerNorm(h_dim2)
        self.norm_p1 = nn.LayerNorm(h_dim1)
        self.norm_p2 = nn.LayerNorm(h_dim2)
        self.norm_m1 = nn.LayerNorm(h_dim1)
        self.norm_m2 = nn.LayerNorm(h_dim2)

        self.dropout = nn.Dropout(0.4)
        self.act = nn.LeakyReLU()

    def forward(self, adj_DP, adj_D, adj_P, drug_homo_embed, prot_homo_embed, hete_embed):
        drg1hop = self.dropout(self.act(self.norm_d1(self.wd1(adj_D @ drug_homo_embed))))
        drg2hop = self.dropout(self.act(self.norm_d2(self.wd2(adj_D @ drg1hop))))
        prot1hop = self.dropout(self.act(self.norm_p1(self.wp1(adj_P @ prot_homo_embed))))
        prot2hop = self.dropout(self.act(self.norm_p2(self.wp2(adj_P @ prot1hop))))
        dm_hop1_base = self.dm1(adj_DP @ hete_embed)
        drug_part_h1, protein_part_h1 = torch.split(
            dm_hop1_base, [args.drug_num, args.protein_num], dim=0
        )
        fused_h1_drug = drug_part_h1 + drg1hop
        fused_h1_protein = protein_part_h1 + prot1hop
        dm1hop = self.dropout(self.act(self.norm_m1(
            torch.cat([fused_h1_drug, fused_h1_protein], dim=0)
        )))
        dm_hop2_base = self.dm2(adj_DP @ dm1hop)
        drug_part_h2, protein_part_h2 = torch.split(
            dm_hop2_base, [args.drug_num, args.protein_num], dim=0
        )
        fused_h2_drug = drug_part_h2 + drg2hop
        fused_h2_protein = protein_part_h2 + prot2hop
        dm2hop = self.dropout(self.act(self.norm_m2(
            torch.cat([fused_h2_drug, fused_h2_protein], dim=0)
        )))
        return drg1hop, drg2hop, prot1hop, prot2hop, dm1hop, dm2hop


class Neighbor_info_integration(nn.Module):
    def __init__(self):
        super().__init__()
        self.ffn_d = FFN_homo_drug()
        self.ffn_p = FFN_homo_prot()
        self.ffn_dm = FFN_hete()
        self.act = nn.LeakyReLU()

    def forward(self, drg1hop, drg2hop, prot1hop, prot2hop, dm1hop, dm2hop, x1, x2):
        drg_embed = torch.cat([drg1hop, drg2hop], dim=1)
        prot_embed = torch.cat([prot1hop, prot2hop], dim=1)
        dm_embed = torch.cat([dm1hop, dm2hop], dim=1)
        drg_embed = self.ffn_d(drg_embed)
        prot_embed = self.ffn_p(prot_embed)
        dm_embed = self.ffn_dm(dm_embed)

        embed_hete_pair = torch.cat([
            dm_embed[x1][:, None, None, :],
            dm_embed[x2 + args.drug_num][:, None, None, :],
        ], dim=2)
        embed_homo_pair = torch.cat([
            drg_embed[x1][:, None, None, :],
            prot_embed[x2][:, None, None, :],
        ], dim=2)
        return torch.cat([embed_homo_pair, embed_hete_pair], dim=3)
