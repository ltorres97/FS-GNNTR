
import torch
from torch import nn, einsum
from torch_geometric.nn import global_mean_pool
from gnn_models import GNN
#from torchmeta.modules import (MetaModule, MetaSequential, MetaConv1d,
                               #MetaBatchNorm1d, MetaLinear)
from torch_geometric.nn import MessagePassing
from einops import rearrange
from torch_geometric import utils
import torch_geometric.utils as utils
from torch_scatter import scatter_add, scatter_mean, scatter_max
import torch_geometric.nn as gnn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch.nn.functional as F
import math
from vit_pytorch.vit import Transformer, ViT

from mlxtend.plotting import heatmap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import seaborn 

num_atom_type = 120 #including the extra mask tokens
num_chirality_tag = 3

num_bond_type = 6 #including aromatic and self-loop edge, and extra masked tokens
num_bond_direction = 3 


def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def position_emb(patches, temperature = 10000, dtype = torch.float32):
    _, h, w, dim, device, dtype = *patches.shape, patches.device, patches.dtype

    y, x = torch.meshgrid(torch.arange(h, device = device), torch.arange(w, device = device), indexing = 'ij')
    assert (dim % 4) == 0, 'feature dimension must be multiple of 4 for sincos emb'
    omega = torch.arange(dim // 4, device = device) / (dim // 4 - 1)
    omega = 1. / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :] 
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim = 1)
    return pe.type(dtype)

# classes

class FeedForwardLayer(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
    def forward(self, x):
        return self.net(x)

class AttentionLayer(nn.Module):
    def __init__(self, dim, heads = 5, dim_head = 64):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                AttentionLayer(dim, heads = heads, dim_head = dim_head),
                FeedForwardLayer(dim, mlp_dim)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class TR(nn.Module):
    def __init__(self, emb_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels = 1, dim_head = 64):
        super().__init__()
        emb_height, emb_width = pair(emb_size)
        patch_height, patch_width = pair(patch_size)

        n_patches = (emb_height // patch_height) * (emb_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b h w (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)

        self.to_latent = nn.Identity()
        self.linear_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, emb):
        *_, h, w, dtype = *emb.shape, emb.dtype
        
        emb = emb.reshape(10,1,300,1)
        
        h = self.to_patch_embedding(emb)
        pe = position_emb(h)
        h = rearrange(h, 'b ... d -> b (...) d') + pe

        h = self.transformer(h)
        h = h.mean(dim = 1)

        h = self.to_latent(h)
        return self.linear_head(h), h
    
class GNN_prediction(torch.nn.Module):

    def __init__(self, layer_number, emb_dim, jk = "last", dropout_prob= 0, pooling = "mean", gnn_type = "gin"):
        super(GNN_prediction, self).__init__()
        
        self.num_layer = layer_number
        self.drop_ratio = dropout_prob
        self.jk = jk
        self.emb_dim = emb_dim

        if self.num_layer < 2:
            raise ValueError("Number of layers must be > 2.")

        self.gnn = GNN(layer_number, emb_dim, jk, dropout_prob, gnn_type = gnn_type)
        
        if pooling == "mean":
            self.pool = global_mean_pool
        else:
            raise ValueError("Invalid pooling.")

        self.mult = 1
        self.graph_pred_linear = torch.nn.Linear(self.mult * self.emb_dim, 1)
        
    def from_pretrained(self, model_file):
        self.gnn.load_state_dict(torch.load(model_file, map_location='cuda:0'), strict = False)
        
    def forward(self, *argv):
        if len(argv) == 4:
            x, edge_index, edge_attr, batch = argv[0], argv[1], argv[2], argv[3]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        else:
            raise ValueError("The arguments are unmatched!")

        node_embeddings = self.gnn(x, edge_index, edge_attr)
            
        pred_gnn = self.graph_pred_linear(self.pool(node_embeddings, batch))
        
        return pred_gnn, node_embeddings
        
        
if __name__ == "__main__":
    pass
