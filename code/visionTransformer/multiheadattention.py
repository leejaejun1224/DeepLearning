import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from Patchembedding import PatchEmbedding

def clones(module, N):
    return nn.ModuleList(copy.deepcopy(module) for _ in range(N))


def scaled_dot_product_attention(query, key, value, mask = None):
    d_k = query.size(-1)
    matmul_qk = torch.matmul(query, key.transpose(-2,-1))
    attn_score = matmul_qk / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
    if mask is not None:
        mask = mask

    attn_score = F.softmax(attn_score, dim=-1)
    attn_value = torch.matmul(attn_score, value)
    return attn_value, attn_score

# 내가 attention head를 몇 개를 줄까
# 여기에서 emb_dim은 channel * 패치하나당 픽셀의 수* 패치하나당 픽셀의 수
# len_seq는 뭐라고 정의할 수 있을까
class MultiheadAttention(nn.Module):    
    def __init__(self, emb_dim, num_heads):
        super(MultiheadAttention, self).__init__()
        assert emb_dim%num_heads == 0
        self.num_heads = num_heads
        self.d_k = emb_dim//num_heads
        self.emb_dim = emb_dim
        # 쪼개서 학습을 하자

        self.weights = clones(nn.Linear(emb_dim, emb_dim), 4)
    
    def forward(self, query, key, value, mask =None):

        n_batches = query.size(0)

        query, key, value = [l(x).view(n_batches, -1, self.num_heads, self.d_k).transpose(1,2) for l, x in zip (self.weights, (query, key, value))]

        attn_value, attn_score = scaled_dot_product_attention(query, key, value, mask)

        attn_value = attn_value.permute(0,2,1,3)

        # concat
        concat_attn_value = attn_value.reshape(n_batches, -1, self.emb_dim)

        output = self.weights[-1](concat_attn_value)

        return output


# if __name__=="__main__":
#     img = torch.randn([2,3,32,32])
#     embedding = PatchEmbedding(channel=3, emb_dim=192, patch_size=4, img_size=32)
#     z = embedding(img)

#     mha = MultiheadAttention(emb_dim=192, num_heads=8)
#     output = mha(z, z, z, mask=None)

#     print(output.size())
#     torch.Size([2, 65, 192])


