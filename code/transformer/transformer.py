
import torch
import torch.nn as nn
from encoder import Encoder, EncoderLayer
from decoder import Decoder, DecoderLayer
from mask import create_look_ahead_mask, create_padding_mask


class Transformer(nn.Module):
    def __init__(self, d_model, num_heads, dff, num_vocabs, dropout, num_layers):
        super(Transformer, self).__init__()

        self.encoder = Encoder(num_vocabs, d_model, num_layers, num_heads, dff, dropout)
        self.decoder = Decoder(num_vocabs, d_model, num_layers, num_heads, dff, dropout)
        self.final_layer = nn.Linear(d_model, num_vocabs)

    def forward(self, enc_input, dec_input):
        enc_mask = create_padding_mask(enc_input)
        dec_mask = create_padding_mask(enc_input)
        enc_output = self.encoder(enc_input, enc_mask)
        dec_output = self.decoder(dec_input, enc_output, dec_mask)
        final_output = self.final_layer(dec_output)
        return final_output
    
