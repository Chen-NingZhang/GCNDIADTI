import argparse
import torch


def parse_args():
    parser = argparse.ArgumentParser(description='GCNDIADTI Model Parameters')

    parser.add_argument('--latdim', default=2220, type=int)
    parser.add_argument('--drug_num', default=708, type=int)
    parser.add_argument('--protein_num', default=1512, type=int)

    parser.add_argument('--gnn_layer', default=3, type=int)
    parser.add_argument('--droprate', default=0.5, type=float)
    parser.add_argument('--node_type_dim', default=32, type=int)
    parser.add_argument('--hyperNum', default=64, type=int)

    parser.add_argument('--attention_heads', default=2, type=int)
    parser.add_argument('--head_dim', default=64, type=int)
    parser.add_argument('--embed_dropout', default=0.3, type=float)
    parser.add_argument('--attention_dropout', default=0.3, type=float)
    parser.add_argument('--depth_interact_attention', default=1, type=int)
    parser.add_argument('--mlp_dim', default=512, type=int)

    parser.add_argument('--label_smoothing', default=0.1, type=float)
    parser.add_argument('--device',
                        default='cuda:0' if torch.cuda.is_available() else 'cpu',
                        type=str)

    return parser.parse_args()


args = parse_args()
