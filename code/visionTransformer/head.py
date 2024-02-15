import torch
import torch.nn as nn

class Head(nn.Module):
    def __init__(self, emb_dim, out_dim):
        super(Head, self).__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(emb_dim),
            #outdim is for number of class
            nn.Linear(emb_dim, out_dim)
        )

    def forward(self, x):
        x = self.head(x)

        return x
    



