import torch
import torch.nn as nn
from einops import repeat
from Patchembedding import PatchEmbedding
from encoder import Encoder
from head import Head
class VisionTransformer(nn.Module):
    def __init__(self, channel, emb_dim, patch_size, img_size, batch_size, num_patches, num_heads, dropout, dff, num_encoder, out_dim):
        super(VisionTransformer, self).__init__()

        # 1. Patch embedding & Linear Projection
        self.patch_embedding = PatchEmbedding(channel, emb_dim, patch_size, img_size)

        # 2. positional embedding
        # self.positional_embedding = nn.Parameter(torch.randn(1, num_patches + 1, emb_dim))

        # 3. cls token
        # self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim))

        # 4. encoder
        self.encoder = Encoder(num_encoder, emb_dim, num_heads, dff, dropout)

        # 5. head
        self.head = Head(emb_dim, out_dim)

    def forward(self, x):
        x = self.patch_embedding(x)
        
        x = self.encoder(x)

        output = self.head(x)

        return output








