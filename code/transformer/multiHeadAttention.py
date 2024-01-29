import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
import matplotlib.pyplot as plt
from torch.autograd import Variable

from mask import create_look_ahead_mask

#4개의 다른 선형 변환을 수행하기 위해서 4개의 fc layer를 생성해줌 
# 대신 input은 같음. 아래에 나옴
def clones(module, N):
        return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

# def attention(query, key, value, mask=None, dropout=None):
#     d_k = query.size(-1)
#     # 첫 번째 차원의 경우 batch_size가 들어가는 건가
#     # 여기서 query, key, value는 W가 각각 곱해진 query, key, value가 되는 것임.
#     scores = torch.matmul(query, key.transpose(-2, -1))/math.sqrt(d_k)

#     if mask is not None:
#         # 0인 부분은 모두 엄청나게 작은 수로 만들어버림
#         # -1 x 10의 9승 -> -10억
#         socres = socres.masked_fill(mask==0, -1e9)

#     # 맨 마지막 차원에 대해서만 softmax함수를 취함. 즉 query @ key해서 나온 내적의 결과는
#     # 각 쿼리마다 key랑 연관성일텐에 각 쿼리마다(단어마다) 한 행을 이룸.
#     p_attn = F.softmax(scores, dim=-1)
#     if dropout is not None:
#         p_attn = dropout(p_attn)

#     return torch.matmul(p_attn, value), p_attn

def scaled_dot_product_attention(query, key, value, mask=None):
    
    #query, key, value의 크기는 모두 batch_size, attention head갯수, 문장에서 단어의 갯수, d_model/attention_head의 갯수
    matmul_qk = torch.matmul(query, key.transpose(-2, -1))
    depth = query.size(-1)

    attentionscore = matmul_qk / torch.sqrt(torch.tensor(depth, dtype=torch.float32))


    if mask is not None:
        attentionscore += (mask * -1e9)

    attentionscore = F.softmax(attentionscore, dim=-1)
    
    output = torch.matmul(attentionscore, value)

    return output, attentionscore

class MultiHeadAttention(nn.Module):
    #여기서 h는 head의 갯수를 의미, 
    def __init__(self, num_head, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        # 뒤에가 True가 아니면 에러 발생시킴
        # 여기서 d_model은 input이 있고 이를 임베딩했을 때의 크기를 말함.
        assert d_model%num_head == 0
        self.d_model = d_model
        # 그냥 뭐 키의 갯수이지 뭐 문장 하나를 학습시키기 위해서 
        # 설정한 임베딩의 차원이라고 볼 수 있음 논문에선 512로 되어있는 것으로 알고있음
        self.d_k = d_model//num_head
        #attention head의 수
        self.num_head = num_head
        # layer를 n개만큼 복사 여기선 4개. 1개는 skip connection, 나머지 3개는 multiheadattention수행
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        
        # 이 부분 이해 못함
        if mask is not None:
            mask = mask.unsqueeze(1)

        # 근데 이렇게 되어있으면 한 문장에서 단어의 갯수를 의미하는게 아닌가?
        # 아님 여기에선 몇 개의 문장을 넣을 것인가를 말함.
        nbatches = query.size(0)

        # 여기는 embedding된 input벡터(x)에서 쿼리 키 밸류로 바꾸는 과정임
        # 실제로 zip의 query key value에서 각 부분은 이따가 나오겠지만 모두 x가 들어감 아직 각 query, key value로 변환 전임
        query, key, value = [l(x).view(nbatches, -1, self.num_head, self.d_k).transpose(1,2) 
                                                    for l, x in zip(self.linears, (query, key, value))]
        # self.linears는 선형 변환을 수행하는 4개의 레이어가 담긴 리스트
        # 여기서 먼저 3개의 선형변환 행렬(가중치)에서 각각 query, key, value를 만듬
        # 처음에 일단 x들에 대해서 (d_model * d_model)의 가중치에 곱해서 (len_seq * d_model)의 행렬을 만든 다음에 이걸
        # 8개로 쪼개서
        # 여기서 헷갈리면 안되는데 x에서 쪼개고 가중치에 곱해지는 것이 아니라 x에서 가중치를 (d_model * d_model) 이 만큼 곱한 다음에 
        # 쪼개는 것이기 때문에 input인 x의 정보 즉 단어를 쪼개서 네트워크에 넣는 것이 아니다는 것을 헷갈리지 말자.
        x, self.attn = scaled_dot_product_attention(query, key, value)
        #아직 코드 미완성

        x = x.permute(0,2,1,3)

        concat_attention = x.reshape(nbatches, -1, self.d_model)

        output = self.linears[-1](concat_attention)

        return output