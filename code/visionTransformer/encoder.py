import torch
import torch.nn as nn
from multiheadattention import MultiheadAttention

class Encoderlayer(nn.Module):
    def __init__(self, d_model, num_heads, dff, dropout):
        super(Encoderlayer,self).__init__()
        
        self.mha = MultiheadAttention(d_model, num_heads)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, dff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dff, d_model),
            nn.Dropout(dropout)
        )

        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, x):

        normoutput1 = self.layernorm1(x)
        output1 = self.mha(normoutput1,normoutput1,normoutput1)
        output1 = x + output1

        normoutput2 = self.layernorm2(output1)
        output2 = self.mlp(normoutput2)
        output = output1 + output2

        return output



class Encoder(nn.Module):
    def __init__(self, num_encoder, d_model, num_heads, dff, dropout):
        super(Encoder, self)
        self.embedding = []
        self.pos_encoding = []

        self.num_encoder = num_encoder
        self.encoder_layer = nn.ModuleList([Encoderlayer(d_model, num_heads, dff, dropout) for _ in range(self.num_encoder)])
    
    def forward(self, x):

        for i in range(self.num_encoder):
            x = self.encoder_layer[i](x)

        return x




