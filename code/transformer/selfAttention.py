import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

np.set_printoptions(suppress=True)

def scaled_dot_product_attention(query, key, value, mask):
    
    #query, key, value의 크기는 모두 batch_size, attention head갯수, 문장에서 단어의 갯수, d_model/attention_head의 갯수
    matmul_qk = torch.matmul(query, key.transpose(-2, -1))
    depth = query.size(-1)

    attentionscore = matmul_qk / torch.sqrt(torch.tensor(depth, dtype=torch.float32))


    if mask is not None:
        attentionscore += (mask * -1e9)

    attentionscore = F.softmax(attentionscore, dim=-1)
    
    output = torch.matmul(attentionscore, value)

    return output, attentionscore


## for testing
temp_k = torch.tensor([[10,0,0],
                      [0,10,0],
                      [0,0,10],
                      [0,0,10]], dtype=torch.float32)  # (4, 3)
temp_v = torch.tensor([[   1,0],
                      [  10,0],
                      [ 100,5],
                      [1000,6]], dtype=torch.float32)  # (4, 2)
temp_q = torch.tensor([[0, 10, 0]], dtype=torch.float32)  # (1, 3)
temp_q = torch.tensor([[0, 0, 10], [0, 10, 0], [10, 10, 0]], dtype=torch.float32)  # (3, 3)
temp_out, temp_attn = scaled_dot_product_attention(temp_q, temp_k, temp_v, None)
print(temp_attn) # 어텐션 분포(어텐션 가중치의 나열)
print(temp_out) # 어텐션 값