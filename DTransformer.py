import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, dim, depth, heads, mlp_dim, channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        x += self.pos_embedding[:, :n]
        x = self.dropout(x)

        x = self.transformer(x)

        return x
    
    
class ViTWrapper(nn.Module):
    def __init__(self, vit, dim, num_classes):
        super().__init__()
        self.vit = vit
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
    
    def forward(self, img):
        x = self.vit(img)
        x = x.mean(dim = 1)
        return self.mlp_head(x)

class DTransformer(nn.Module):
    def __init__(self, vit, dim, channel_size_lst=[16, 32, 64], div_indices = [5, 10, 15], start_index=None, end_index=None):
        super().__init__()
        self.vit = vit
        decoder_layer = nn.TransformerDecoderLayer(d_model=dim, nhead=8, dim_feedforward=1024, batch_first=True, dropout=0)
        self.decoder = decoder = nn.TransformerDecoder(decoder_layer, num_layers=5)
        # start of sequence token
        self.sos = nn.Parameter(torch.randn(1, 1, dim))
        self.mlp_in = nn.ModuleList()
        self.mlp_out = nn.ModuleList()
        for channel_size in channel_size_lst:
            self.mlp_in.append(nn.Linear(channel_size, dim))
            self.mlp_out.append(nn.Linear(dim, channel_size))
        self.decoder_pe = nn.Parameter(torch.randn(1, div_indices[-1], dim))
        self.div_indices = div_indices
        self.start_index = start_index
        self.end_index = end_index
        
    def pretrain_loss(self, img, teacher):
        context = self.vit(img)
        features = teacher.get_features(img, self.start_index, self.end_index, self.div_indices)
        features_input = self.create_input(features)
        sz = features_input.shape[1]
        # create mask
        mask = (torch.triu(torch.ones((sz, sz), device=context.device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        
        features_output = self.decoder(features_input, context, tgt_mask=mask)
        predict_features = self.create_output(features_output)
        losses = []
        for i in range(len(features)):
            losses.append(nn.functional.mse_loss(predict_features[i], features[i], reduction='none').flatten())
        return torch.mean(torch.cat(losses))
    
    def create_input(self, features):
        tokens = []
        for i in range(len(features)):
            tokens.append(self.mlp_in[i](features[i]))
        tokens = torch.cat(tokens, dim=1)
        tokens = tokens[:, :-1, :]
        b = tokens.shape[0]
        sos = repeat(self.sos, '1 n d -> b n d', b = b)
        tokens = torch.cat((sos, tokens), dim=1)
        tokens += self.decoder_pe
        return tokens
    
    def create_output(self, features_output):
        result = []
        start = 0
        for i in range(len(self.div_indices)):
            end = self.div_indices[i]
            result.append(self.mlp_out[i](features_output[:, start:end, :]))
            start = end
        return result
        
        
        
        
        