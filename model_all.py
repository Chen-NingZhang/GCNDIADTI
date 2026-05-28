import torch
import torch.nn as nn

import Model517
from Transformer import InterT
from parameters import args


class final_model(nn.Module):
    def __init__(self, ablation=None):
        super().__init__()
        self.ablation = ablation

        if ablation == 'no_gcn':
            self.no_gcn_drug_proj = nn.Sequential(
                nn.Linear(args.latdim, 512),
                nn.LeakyReLU(),
                nn.Dropout(0.4),
                nn.Linear(512, 256),
                nn.LayerNorm(256),
            )
            self.no_gcn_prot_proj = nn.Sequential(
                nn.Linear(args.latdim, 512),
                nn.LeakyReLU(),
                nn.Dropout(0.4),
                nn.Linear(512, 256),
                nn.LayerNorm(256),
            )
        else:
            self.NAPGCN = Model517.NAPGCN()
            self.neighbor_info_integration = Model517.Neighbor_info_integration()

        self.transformer = InterT()

        self.gnn_context_projection = nn.Linear(256, 256)

        self.gate_drug = nn.Sequential(nn.Linear(256 * 2, 256), nn.Sigmoid())
        self.gate_prot = nn.Sequential(nn.Linear(256 * 2, 256), nn.Sigmoid())

        self.fusion_linear = nn.Linear(256, 512)
        self.fusion_norm = nn.LayerNorm(512)
        self.fusion_act = nn.LeakyReLU()
        self.fusion_dropout = nn.Dropout(0.4)

        self.classifier = Model517.MLPClassifier()

        self.temperature = nn.Parameter(torch.ones(1) * 1.0)

    def forward(self, x1, x2, embeds, adj_DP, adj_D, adj_P):
        if self.ablation == 'no_gcn':
            neigh_drug_features = self.no_gcn_drug_proj(embeds[x1])
            neigh_prot_features = self.no_gcn_prot_proj(embeds[x2 + args.drug_num])
        else:
            hete_embed = embeds
            drug_homo_embed = embeds[:args.drug_num, :]
            prot_homo_embed = embeds[args.drug_num:args.latdim, :]

            drg1hop, drg2hop, prot1hop, prot2hop, dm1hop, dm2hop = self.NAPGCN(
                adj_DP, adj_D, adj_P, drug_homo_embed, prot_homo_embed, hete_embed
            )

            x_neighbor = self.neighbor_info_integration(
                drg1hop, drg2hop, prot1hop, prot2hop, dm1hop, dm2hop, x1, x2
            )

            neigh_drug_features = x_neighbor[:, 0, 0, :]
            neigh_prot_features = x_neighbor[:, 0, 1, :]

        drug_gnn_context = self.gnn_context_projection(neigh_drug_features).unsqueeze(1)
        protein_gnn_context = self.gnn_context_projection(neigh_prot_features).unsqueeze(1)

        x_transformer = self.transformer(
            neigh_drug_features, neigh_prot_features,
            drug_gnn_context, protein_gnn_context,
        )
        trans_drug, trans_prot = x_transformer[:, 0, 0, :], x_transformer[:, 0, 1, :]

        g_drug = self.gate_drug(torch.cat([neigh_drug_features, trans_drug], dim=1))
        g_prot = self.gate_prot(torch.cat([neigh_prot_features, trans_prot], dim=1))

        fused_drug = neigh_drug_features + g_drug * trans_drug
        fused_prot = neigh_prot_features + g_prot * trans_prot

        fused_drug = self.fusion_dropout(self.fusion_act(
            self.fusion_norm(self.fusion_linear(fused_drug))
        ))
        fused_prot = self.fusion_dropout(self.fusion_act(
            self.fusion_norm(self.fusion_linear(fused_prot))
        ))

        x = torch.cat([fused_drug, fused_prot], dim=1)
        x = x.unsqueeze(1).unsqueeze(1)

        output = self.classifier(x)

        output = output / self.temperature

        return output
