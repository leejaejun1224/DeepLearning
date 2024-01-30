import torch
import torch.nn as nn
from multiHeadAttention import MultiHeadAttention
from positionalencoding import PositionalEncoding

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dff, dropout):
        super(EncoderLayer, self).__init__()
        # multiheadattention
        self.mha = MultiHeadAttention(num_heads, d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dff),
            nn.ReLU(),
            nn.Linear(dff, d_model)
        )

        # d_model이라고 입력을 해주면 알아서 d_model의 차원에 맞는 행이나 열을 가져다가 정규화를 해줌 신기함
        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)

        # 무작위로 뉴런을 꺼버린다. 사용 방식을 포워드에 기술
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, padding_mask):
        # print("encoding")

        attn_output = self.mha(x,x,x,padding_mask)

        #사실 기술 할 것도 없음. 그냥 랜덤으로 0으로 만들어버리는 것 뿐
        attn_output = self.dropout1(attn_output)
        #skip connection(residual)
        output1 = self.layernorm1(x + attn_output)

        ffn_out = self.ffn(output1)
        ffn_out = self.dropout2(ffn_out)
        
        encoder_output = self.layernorm2(ffn_out + output1)

        return encoder_output


class Encoder(nn.Module):
    def __init__(self, num_vocabs,d_model, num_layers, num_heads, dff, dropout):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = nn.Embedding(num_vocabs, d_model)

        #이렇게 하면 너무 커지는데 우리는 len_seq만 알면 되는데
        self.pos_encoding = PositionalEncoding(num_vocabs, d_model)
        self.encoderlayer = nn.ModuleList([EncoderLayer(d_model, num_heads, dff, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask): 

        x = self.embedding(x)
        # print("size of embedding tensor", x.shape)
        x *= torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        x = self.pos_encoding(x)
        x = self.dropout(x)

        # print("encoder mask shape :", mask.shape)
        for i in range(self.num_layers):
            x = self.encoderlayer[i](x, mask)

        return x


