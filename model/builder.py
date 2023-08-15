import torch
import torch.nn as nn
from .models import *
from .backbones import *
from .msdeform_decoder import VLMSDeformAttnPixelDecoder

def _segm_caris(pretrained, args, criterion):
    # initialize the SwinTransformer backbone with the specified version
    if args.swin_type == 'tiny':
        embed_dim = 96
        depths = [2, 2, 6, 2]
        num_heads = [3, 6, 12, 24]
    elif args.swin_type == 'small':
        embed_dim = 96
        depths = [2, 2, 18, 2]
        num_heads = [3, 6, 12, 24]
    elif args.swin_type == 'base':
        embed_dim = 128
        depths = [2, 2, 18, 2]
        num_heads = [4, 8, 16, 32]
    elif args.swin_type == 'large':
        embed_dim = 192
        depths = [2, 2, 18, 2]
        num_heads = [6, 12, 24, 48]
    else:
        assert False
    # args.window12 added for test.py because state_dict is loaded after model initialization
    if 'window12' in pretrained or args.window12:
        print('Window size 12!')
        window_size = 12
    else:
        window_size = 7

    out_indices = (0, 1, 2, 3)
    channels = [embed_dim, embed_dim*2, embed_dim*4, embed_dim*8]
    prompt_levels = [1, 2]
    backbone = PromptEncoder(args, channels, prompt_levels=prompt_levels, embed_dim=embed_dim, depths=depths, num_heads=num_heads,
                                            window_size=window_size,
                                            ape=False, drop_path_rate=0.3, patch_norm=True,
                                            out_indices=out_indices, use_checkpoint=False
                                            )

    if pretrained:
        print('Initializing Multi-modal Swin Transformer weights from ' + pretrained)
        backbone.vis_encoder.init_weights(pretrained=pretrained)
    else:
        print('Randomly initialize Multi-modal Swin Transformer weights.')
        backbone.vis_encoder.init_weights()

    d_model = 256
    num_classes = 2
    num_vis_prompts = 5
    pixel_decoder = VLMSDeformAttnPixelDecoder(in_channels=channels, feat_channels=d_model, out_channels=d_model, 
                                               num_enc_layers=6, num_heads=8, im2col_step=16, num_points=4, num_levels=3,
                                               mlp_expand=4, dropout=0, 
                                               with_prompts=True, num_prompts=num_vis_prompts)
    model = CARIS(backbone, pixel_decoder, args, num_classes=num_classes, criterion=criterion)
    return model

def caris(pretrained='', args=None, criterion=None):
    return _segm_caris(pretrained, args, criterion)