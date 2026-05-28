import torch
import torch.nn as nn
from einops import rearrange

from parameters import args


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        ) if project_out else nn.Identity()

    def forward(self, x, context=None):
        x = self.norm(x)
        context = x if context is None else context

        q = self.to_q(x)
        k, v = self.to_kv(context).chunk(2, dim=-1)

        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.heads)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class InteractionBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.1):
        super().__init__()
        self.drug_self_attn = Attention(dim, heads, dim_head, dropout)
        self.protein_self_attn = Attention(dim, heads, dim_head, dropout)
        self.drug_self_ffn = FeedForward(dim, mlp_dim, dropout)
        self.protein_self_ffn = FeedForward(dim, mlp_dim, dropout)

        self.drug_cross_attn = Attention(dim, heads, dim_head, dropout)
        self.protein_cross_attn = Attention(dim, heads, dim_head, dropout)
        self.drug_cross_ffn = FeedForward(dim, mlp_dim, dropout)
        self.protein_cross_ffn = FeedForward(dim, mlp_dim, dropout)

    def forward(self, x_drug, x_protein, drug_gnn_context, protein_gnn_context):
        x_drug = self.drug_self_attn(x_drug) + x_drug
        x_drug = self.drug_self_ffn(x_drug) + x_drug

        x_protein = self.protein_self_attn(x_protein) + x_protein
        x_protein = self.protein_self_ffn(x_protein) + x_protein

        drug_cross_out = self.drug_cross_attn(x_drug, context=protein_gnn_context) + x_drug
        protein_cross_out = self.protein_cross_attn(x_protein, context=drug_gnn_context) + x_protein

        drug_final = self.drug_cross_ffn(drug_cross_out) + drug_cross_out
        protein_final = self.protein_cross_ffn(protein_cross_out) + protein_cross_out

        return drug_final, protein_final


class InterT(nn.Module):
    def __init__(self):
        super().__init__()
        transformer_dim = 256
        gcn_out_dim = 256

        self.input_projection = nn.Linear(gcn_out_dim, transformer_dim)

        self.dropout = nn.Dropout(args.embed_dropout + 0.1)

        self.transformer_layers = nn.ModuleList([])
        for _ in range(args.depth_interact_attention):
            self.transformer_layers.append(InteractionBlock(
                dim=transformer_dim,
                heads=args.attention_heads,
                dim_head=args.head_dim,
                mlp_dim=args.mlp_dim,
                dropout=args.attention_dropout,
            ))

        self.output_norm = nn.LayerNorm(transformer_dim)

    def forward(self, h_d, h_t, drug_gnn_context, protein_gnn_context):
        x_drug = self.input_projection(h_d).unsqueeze(1)
        x_protein = self.input_projection(h_t).unsqueeze(1)

        x_drug = self.dropout(x_drug)
        x_protein = self.dropout(x_protein)

        for layer in self.transformer_layers:
            x_drug, x_protein = layer(x_drug, x_protein,
                                      drug_gnn_context, protein_gnn_context)

        x_drug_final = self.output_norm(x_drug.squeeze(1))
        x_protein_final = self.output_norm(x_protein.squeeze(1))

        x_transformer = torch.stack(
            [x_drug_final, x_protein_final], dim=1
        ).unsqueeze(1)
        return x_transformer
