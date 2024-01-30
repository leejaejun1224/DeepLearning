import torch

def create_padding_mask(x):
    # 0으로 되어있는 곳만 1이고 나머지가 다 0인 tensor를 반환함.
    mask = torch.eq(x, 0).float()
    # (batch_size, 1, 1, key의 문장 길이)
    return mask.unsqueeze(1).unsqueeze(2)

array = torch.tensor([[1.0,0.0],[0.5,0.8]])
# print(create_padding_mask(array).shape)

# def create_look_ahead_mask(x, padding_mask):
#     seq_len = x.size(1)
#     initarray = torch.ones((seq_len, seq_len))
#     look_ahead_mask = torch.triu(initarray, diagonal=1)
#     # print("패딩마스크의 크기 : ", padding_mask.shape)
#     # print("룩어헤드 마스크의 크기 : ",look_ahead_mask.shape)
#     #그냥 padding이랑 look ahead padding둘 다 필요
#     return look_ahead_mask.unsqueeze(0)

def create_look_ahead_mask(x, batch_size):
    # 상삼각 행렬 생성 (대각선을 포함하지 않음)
    size = x.size(1)
    mask = torch.triu(torch.ones(size, size), diagonal=1)
    # 배치 차원 추가 ([batch_size, 1, seq_len, seq_len])
    mask = mask.unsqueeze(0).repeat(batch_size, 1, 1, 1)
    return mask
# def create_look_ahead_mask(x, padding_mask):
#     seq_len = x.size(1)
#     look_ahead_mask = 1 - torch.tril(torch.ones((seq_len, seq_len)))
#     padding_mask = create_padding_mask(x) # 패딩 마스크도 포함
#     torch.max(look_ahead_mask, padding_mask).shape
#     return torch.max(look_ahead_mask, padding_mask)

array = torch.tensor([[1, 2, 0, 4, 5]], dtype=torch.float32)

# print(create_look_ahead_mask(array))
# print(create_padding_mask(array))
