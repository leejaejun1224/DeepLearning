import torch
import torch.nn as nn
from torch import Tensor
from einops.layers.torch import Rearrange
from einops import repeat
### for patch flatten and linear projection


class PatchEmbedding(nn.Module):
    def __init__(self, channel=3, emb_dim = 128, patch_size = 16, img_size = 224):
        super(PatchEmbedding,self).__init__()

        h = img_size // patch_size
        w = img_size // patch_size
        self.embedding = nn.Sequential(
            # fucking korean why not working sibal
            # same as reshape what a creative funtcions
            # height automatically divde into h, p1 / width automatically divde into w, p2
            # we call h*w as N in paper
            # and we need to plus padding mask

            # h = height / patch_size, w = width / patch_size
            Rearrange('b c (h p1) (w p2) -> b (h w) (c p1 p2)', p1 = patch_size, p2 = patch_size),
            nn.Linear(patch_size*patch_size*channel, emb_dim)
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, h*w+ 1, emb_dim))


    def forward(self, x: Tensor):
        x = self.embedding(x)
        batch_size = x.size(0)
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = batch_size)
        x = torch.cat([cls_tokens, x], dim=1)
        x += self.pos_embedding
        return x


# # 확인용
# if __name__=="__main__":
#     img = torch.randn([2,3,32,32])
#     embedding = PatchEmbedding(channel=3, emb_dim=192, patch_size=4, img_size=32)
#     z = embedding(img)
#     print(z.size())


