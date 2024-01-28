import torch
import torch.nn as nn

class FFNN(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.lienar = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()


    def forward():
        ...

# 굳이?