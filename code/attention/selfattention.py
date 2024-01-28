import torch
import torch.nn as nn
from torch.nn.functional import softmax

input = [[1,0,1,0],[0,2,0,2],[1,1,1,1]]

w_query = [[1,0,1],[1,0,0],[0,0,1],[0,1,1]]
w_key = [[0,0,1],[1,1,0],[0,1,0],[1,1,0]]
w_value = [[0,2,0],[0,3,0],[1,0,3],[1,1,0]]

w_query = torch.tensor(w_query, dtype=torch.float32)
w_key = torch.tensor(w_key, dtype=torch.float32)
w_value = torch.tensor(w_value, dtype=torch.float32)

def attentionScore(x):
    x = torch.tensor(x, dtype=torch.float32)
    query = x@w_query
    key = x@w_key
    value = x@w_value

    score = query @ key.T
    return score

input = torch.tensor(input, dtype=torch.float32)

querys = input@w_query
keys = input@w_key
values = input@w_value

attention_score = querys @ keys.T

attention_score_softmax_approx = (softmax(attention_score, dim=-1)*10).floor()/10

# 이렇게 두 번째에 None이 있으면 두 번째에 하나의 차원을 욱여넣는 것
# 마지막에 있으면 마지막에 하나의 차원을 욱여 넣는 것
weighted_value = values[:,None] * attention_score_softmax_approx.T[:,:,None]
# print(values[:,None])
print(attention_score_softmax_approx.T[:,:,None])
# print(weighted_value)
# output = weighted_value.sum(dim=0)
# print("output = ", output)

print(values[:,None]*attention_score_softmax_approx.T[:,:,None][0])

