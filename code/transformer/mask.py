import torch

def create_padding_mask(x):
    # 0으로 되어있는 곳만 1이고 나머지가 다 0인 tensor를 반환함.
    mask = torch.eq(x, 0).float()
    # (batch_size, 1, 1, key의 문장 길이)
    return mask.unsqueeze(1).unsqueeze(2)

array = torch.tensor([[1.0,0.0],[0.5,0.8]])
# print(create_padding_mask(array).shape)

def create_look_ahead_mask(x):
    seq_len = x.size(1)
    initarray = torch.ones(seq_len, seq_len)
    look_ahead_mask = torch.triu(initarray, diagonal=1)
    return look_ahead_mask*-1e9

print(create_look_ahead_mask(array))
