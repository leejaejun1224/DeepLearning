import torch
import torch.nn as nn
# 가정: d_model = 512, 임베딩 차원
d_model = 512
# 임의의 임베딩 벡터 생성 (예: 배치 크기 64, 시퀀스 길이 20)
batch_size = 1
seq_length = 4
embedding = torch.randn(batch_size, seq_length, d_model)

# 임베딩 스케일링
scaling_factor = torch.sqrt(torch.tensor(d_model, dtype=torch.float32))
scaled_embedding = embedding * scaling_factor

print("원래 임베딩 크기:", embedding.size())
print("스케일링 후 임베딩 크기:", scaled_embedding.size())
print(embedding)
print(scaled_embedding)

embedding_layer = nn.Embedding(10000, 512)
input_word = torch.tensor([45])  # '안녕'의 정수 인코딩 값
embedded_word = embedding_layer(input_word)
print(embedded_word)
