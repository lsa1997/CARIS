import torch
from torch import nn
from torch.nn import functional as F
from .swin import SwinTransformer

class MultiStageSwinTransformer(SwinTransformer):
    '''
        SwinTransformer that supports multi-stage encoding for multimodal learning.
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.n_stages = self.num_layers
        
        assert self.n_stages == 4, 'Only 4-stage index is supported!'
    
    def forward_embeddings(self, x):
        x = self.patch_embed(x)

        Wh, Ww = x.size(2), x.size(3)
        if self.ape:
            # interpolate the position embedding to the corresponding size
            absolute_pos_embed = F.interpolate(self.absolute_pos_embed, size=(Wh, Ww), mode='bicubic')
            x = (x + absolute_pos_embed).flatten(2).transpose(1, 2)  # [B, Wh*Ww, C]
        else:
            x = x.flatten(2).transpose(1, 2)
        x = self.pos_drop(x)
        return x, Wh, Ww

    def forward_stages(self, x, Wh, Ww, stage, out_norm=False):
        assert stage in self.out_indices, 'stage {} does not exist in out_indices!'.format(stage)

        layer = self.layers[stage]
        x, Wh, Ww = layer.forward_pre_downsample(x, Wh, Ww)
        if out_norm:
            norm_layer = getattr(self, f'norm{stage}')
            x_out = norm_layer(x)  # output of a Block has shape (B, H*W, dim)
            x_out = x_out.view(-1, Wh, Ww, self.num_features[stage]).permute(0, 3, 1, 2).contiguous()
            return x_out, x, Wh, Ww
        else:
            return x, Wh, Ww

    def forward_downs(self, x, Wh, Ww, stage):
        assert stage in self.out_indices, 'stage {} does not exist in out_indices!'.format(stage)

        layer = self.layers[stage]
        x, Wh, Ww = layer.forward_downsample(x, Wh, Ww)

        return x, Wh, Ww

    def forward_norms(self, x, Wh, Ww, stage):
        assert stage in self.out_indices, 'stage {} does not exist in out_indices!'.format(stage)

        norm_layer = getattr(self, f'norm{stage}')
        x_out = norm_layer(x)  # output of a Block has shape (B, H*W, dim)
        x_out = x_out.view(-1, Wh, Ww, self.num_features[stage]).permute(0, 3, 1, 2).contiguous() # [B, dim, Wh, Ww]
        return x_out
