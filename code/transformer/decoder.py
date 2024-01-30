import torch
import torch.nn as nn
import torch.nn.functional as F
from multiHeadAttention import MultiHeadAttention
from mask import create_look_ahead_mask, create_padding_mask
from positionalencoding import PositionalEncoding

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dff, dropout):
        super(DecoderLayer, self).__init__()
        self.d_model = d_model

        self.mmha1 = MultiHeadAttention(num_heads, d_model)
        self.mmha2 = MultiHeadAttention(num_heads, d_model)

        self.ffn1 = nn.Sequential(
            nn.Linear(d_model, dff),
            nn.ReLU(),
            nn.Linear(dff, d_model)
        )

        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm3 = nn.LayerNorm(d_model, eps=1e-6)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, encoder_output, look_ahead_mask, padding_mask):
        # print("decoding")
        mmha_output1 = self.mmha1(x, x, x, look_ahead_mask)
        mmha_output1 = self.dropout1(mmha_output1)
        output1 = self.layernorm1(x + mmha_output1)

        mmha_output2 = self.mmha2(output1, encoder_output, encoder_output, padding_mask)
        mmha_output2 = self.dropout2(mmha_output2)
        output2 = self.layernorm2(output1 + mmha_output2)

        ffn_output = self.ffn1(output2)
        ffn_output = self.dropout3(ffn_output)
        output3 = self.layernorm3(output2 + ffn_output)

        return output3

    

class Decoder(nn.Module):
    def __init__(self, num_vocabs, d_model, num_layers, num_heads, dff, dropout):
        super(Decoder, self).__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        self.embedding = nn.Embedding(num_vocabs, d_model)
        self.pos_encoding = PositionalEncoding(num_vocabs, d_model)
        self.decoder_layer = nn.ModuleList([DecoderLayer(d_model, num_heads, dff, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, padding_mask):
        seq_len = x.size(1)
        x = self.embedding(x)
        x *= torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        x = self.pos_encoding(x)
        x = self.dropout(x)

        look_ahead_mask = create_look_ahead_mask(x, x.size(0))
        for i in range(self.num_layers):
            x = self.decoder_layer[i](x, encoder_output, look_ahead_mask, padding_mask)

        return x
