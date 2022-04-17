from hyperparams import *
import torch
from torch import nn

class ShapeCodeEmbedding(nn.Module):
    def __init__(self):
        super(SDFDecoder, self).__init__()
        
        self.latent_shape_code_emb = nn.Embedding(num_shapes, shape_code_dim)
        # initialize embedding normally
        nn.init.normal_(self.latent_shape_code_emb.weight, mean=0, std=0.01)

    def forward(self, shape_idx):
        return self.latent_shape_code_emb(shape_idx)

class SDFDecoder(nn.Module):
    def __init__(self, encoded_position_dim, hidden_dim=512):
        super(SDFDecoder, self).__init__()
        
        self.mlp_stack = nn.Sequential(
            nn.Linear(encoded_position_dim, hidden_dim).double(),  
            nn.ReLU(),
            nn.Linear(hidden_dim, 1).double(),
        )

    def forward(self, pos):
        # normalize [-1,1]^3 to [0,1]^3
        pos = 0.5*pos + 0.5

        return self.mlp_stack(pos)
