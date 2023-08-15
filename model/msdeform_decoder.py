# Modified from mmdet/models/plugins/msdeformattn_pixel_decoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

from mmcv.cnn import (ConvModule, caffe2_xavier_init, xavier_init)
from mmcv.runner import BaseModule, ModuleList
from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttention
from mmdet.core.anchor import MlvlPointGenerator
from mmdet.models.utils import SinePositionalEncoding

from .tranformer_decoder import FeedForward, MultiHeadAttention, PosEncoding

class VLMSDeformAttnLayer(nn.Module):
    def __init__(self, embed_dim, heads, num_levels, 
                 num_points=4, im2col_step=16, 
                 mlp_expand=4, attn_expand=4,
                 dropout=0., norm_layer=nn.LayerNorm, with_gamma=False, init_value=1.0):
        super().__init__()
        self.vis_self_attn = MultiScaleDeformableAttention(embed_dim, num_heads=heads, num_levels=num_levels, num_points=num_points, 
                                                           im2col_step=im2col_step, dropout=dropout, batch_first=True)
        self.v2l_cross_attn = MultiHeadAttention(embed_dim, embed_dim, embed_dim*attn_expand, num_heads=heads, 
                                                 dropout=dropout, clamp_min_for_underflow=True, clamp_max_for_overflow=True)
        self.l2v_cross_attn = MultiHeadAttention(embed_dim, embed_dim, embed_dim*attn_expand, num_heads=heads, 
                                                 dropout=dropout, clamp_min_for_underflow=True, clamp_max_for_overflow=True)
        self.lang_self_attn = MultiHeadAttention(embed_dim, embed_dim, embed_dim*attn_expand, num_heads=heads, 
                                                 dropout=dropout, clamp_min_for_underflow=True, clamp_max_for_overflow=True)

        self.vis_mlp = FeedForward(embed_dim, embed_dim*mlp_expand, dropout, act='relu')
        self.lang_mlp = FeedForward(embed_dim, embed_dim*mlp_expand, dropout, act='relu')

        self.with_gamma = with_gamma
        if with_gamma:
            self.gamma_v2l = nn.Parameter(init_value*torch.ones((1, 1, embed_dim)), requires_grad=True)
            self.gamma_l2v = nn.Parameter(init_value*torch.ones((1, 1, embed_dim)), requires_grad=True)

        self.v_norm1 = norm_layer(embed_dim)
        self.v_norm2 = norm_layer(embed_dim)
        self.v_norm3 = norm_layer(embed_dim)
        self.v_drop2 = nn.Dropout(dropout)
        self.v_drop3 = nn.Dropout(dropout)

        self.l_norm1 = norm_layer(embed_dim)
        self.l_norm2 = norm_layer(embed_dim)
        self.l_norm3 = norm_layer(embed_dim)
        self.l_drop1 = nn.Dropout(dropout)
        self.l_drop2 = nn.Dropout(dropout)
        self.l_drop3 = nn.Dropout(dropout)
        self.init_weights()

    def extra_repr(self):
        return 'with_gamma={}'.format(self.with_gamma)

    def init_weights(self):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        self.apply(_init_weights)
        self.vis_self_attn.init_weights() # init_weights defined in MultiScaleDeformableAttention

    def forward(self, vis, lang, prompts=None, lang_mask=None, vis_pos=None, vis_padding_mask=None, **kwargs):
        '''
            - vis: :math:`(N, L, E)` where L is the sequence length, N is the batch size, E is
            the embedding dimension.
            - lang: :math:`(N, S, E)`, where S is the sequence length, N is the batch size, E is
            the embedding dimension.
            - prompts: :math:`(N, P, E)` where P is the source sequence length, N is the batch size, E is
            the embedding dimension.
            - lang_mask :math:`(N, S)` where N is the batch size, S is the source sequence length.
            If a ByteTensor is provided, the zero positions will be ignored while the position
            with the non-zero positions will be unchanged.
            - vis_pos: :math:`(N, L, E)` where L is the sequence length, N is the batch size, E is
            the embedding dimension.
            - vis_padding_mask :math:`(N, L)` where N is the batch size, L is the source sequence length.
            If a ByteTensor is provided, the non-zero positions will be ignored while the position
            with the zero positions will be unchanged.
        '''
        if prompts is not None:
            prompted_lang = torch.cat([lang, prompts], dim=1)
        else:
            prompted_lang = lang

        # V2L cross attn
        _prompted_lang = self.v2l_cross_attn(q=prompted_lang, k=PosEncoding(vis, vis_pos), v=vis, attention_mask=(1-vis_padding_mask.byte()))[0]
        if self.with_gamma:
            _prompted_lang = _prompted_lang * self.gamma_v2l
        prompted_lang = prompted_lang + self.l_drop1(_prompted_lang)
        prompted_lang = self.l_norm1(prompted_lang)

        # Linguistic self attn
        _prompted_lang = self.lang_self_attn(q=prompted_lang, k=prompted_lang, v=prompted_lang, attention_mask=lang_mask)[0]
        prompted_lang = prompted_lang + self.l_drop2(_prompted_lang)
        prompted_lang = self.l_norm2(prompted_lang)

        # Linguistic FFN
        _prompted_lang = self.lang_mlp(prompted_lang)
        prompted_lang = prompted_lang + self.l_drop3(_prompted_lang)
        prompted_lang = self.l_norm3(prompted_lang)

        # L2V corss attn
        _vis = self.l2v_cross_attn(q=PosEncoding(vis, vis_pos), k=prompted_lang, v=prompted_lang, attention_mask=lang_mask)[0]
        if self.with_gamma:
            _vis = _vis * self.gamma_l2v
        vis = vis + self.v_drop2(_vis)
        vis = self.v_norm2(vis)

        # Visual MSDeformable self attn
        with torch.cuda.amp.autocast(enabled=False):
            vis = self.vis_self_attn(vis.float(), value=vis.float(), query_pos=vis_pos.float(), key_padding_mask=vis_padding_mask, **kwargs)
        vis = self.v_norm1(vis)

        # Visual FFN
        _vis = self.vis_mlp(vis)
        vis = vis + self.v_drop3(_vis)
        vis = self.v_norm3(vis)

        if prompts is not None:
            lang = prompted_lang[:, :lang.shape[1]]
            prompts = prompted_lang[:, lang.shape[1]:]
        else:
            lang = prompted_lang
            prompts = None
        return vis, lang, prompts
    
class VLMSDeformAttnPixelDecoder(BaseModule):
    """Pixel decoder with multi-scale deformable attention.
    Use learnable prompt, prompt.mean() to predict background
    Args:
        in_channels (list[int] | tuple[int]): Number of channels in the
            input feature maps.
        strides (list[int] | tuple[int]): Output strides of feature from
            backbone.
        feat_channels (int): Number of channels for feature.
        out_channels (int): Number of channels for output.
        num_outs (int): Number of output scales.
        norm_cfg (:obj:`mmcv.ConfigDict` | dict): Config for normalization.
            Defaults to dict(type='GN', num_groups=32).
        act_cfg (:obj:`mmcv.ConfigDict` | dict): Config for activation.
            Defaults to dict(type='ReLU').
        positional_encoding (:obj:`mmcv.ConfigDict` | dict): Config for
            transformer encoder position encoding. Defaults to
            dict(type='SinePositionalEncoding', num_feats=128,
            normalize=True).
    """

    def __init__(self,
                 in_channels=[256, 512, 1024, 2048],
                 lang_in_channels=768,
                 strides=[4, 8, 16, 32],
                 feat_channels=256,
                 out_channels=256,
                 num_enc_layers=6,
                 num_heads=8,
                 num_levels=3,
                 num_points=4,
                 im2col_step=16,
                 dropout=0.0,
                 mlp_expand=4,
                 with_prompts=True,
                 num_prompts=10,
                 norm_cfg=dict(type='GN', num_groups=32),
                 act_cfg=dict(type='ReLU'),
                 ):
        super().__init__()
        self.strides = strides
        self.num_input_levels = len(in_channels)
        self.num_encoder_levels = num_levels
        assert self.num_encoder_levels >= 1, \
            'num_levels in attn_cfgs must be at least one'
        input_conv_list = []
        # from top to down (low to high resolution)
        for i in range(self.num_input_levels - 1,
                       self.num_input_levels - self.num_encoder_levels - 1,
                       -1):
            input_conv = ConvModule(
                in_channels[i],
                feat_channels,
                kernel_size=1,
                norm_cfg=norm_cfg,
                act_cfg=None,
                bias=True)
            input_conv_list.append(input_conv)
        self.input_convs = ModuleList(input_conv_list)
        self.num_enc_layers = num_enc_layers
        self.encoder = nn.ModuleList([VLMSDeformAttnLayer(feat_channels, heads=num_heads, num_levels=num_levels, num_points=num_points,
                                    im2col_step=im2col_step, mlp_expand=mlp_expand, dropout=dropout, with_gamma=True) for i in range(self.num_enc_layers)])
        self.postional_encoding = SinePositionalEncoding(num_feats=feat_channels//2, normalize=True)
        # high resolution to low resolution
        self.level_encoding = nn.Embedding(self.num_encoder_levels, feat_channels)
        # language features
        self.lang_in_linear = nn.Linear(lang_in_channels, feat_channels)
        self.lang_in_norm = nn.LayerNorm(feat_channels)
        # visual prompts
        self.with_prompts = with_prompts
        self.num_prompts = num_prompts
        if with_prompts:
            self.vis_prompts = nn.Embedding(num_prompts, feat_channels)

        # fpn-like structure
        self.lateral_convs = ModuleList()
        self.output_convs = ModuleList()
        self.use_bias = norm_cfg is None
        # from top to down (low to high resolution)
        # fpn for the rest features that didn't pass in encoder
        for i in range(self.num_input_levels - self.num_encoder_levels - 1, -1,
                       -1):
            lateral_conv = ConvModule(
                in_channels[i],
                feat_channels,
                kernel_size=1,
                bias=self.use_bias,
                norm_cfg=norm_cfg,
                act_cfg=None)
            output_conv = ConvModule(
                feat_channels,
                feat_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=self.use_bias,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
            self.lateral_convs.append(lateral_conv)
            self.output_convs.append(output_conv)

        self.point_generator = MlvlPointGenerator(strides)
        self.mask_feature = nn.Sequential(
            nn.Conv2d(feat_channels, feat_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channels, feat_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channels, out_channels, kernel_size=1)
            )
        self.lang_embed = nn.Sequential(
            nn.Linear(feat_channels, feat_channels),
            nn.ReLU(inplace=True),
            nn.Linear(feat_channels, feat_channels),
            nn.ReLU(inplace=True),
            nn.Linear(feat_channels, out_channels)
            )
        
    def init_weights(self):
        """Initialize weights."""
        for i in range(0, self.num_encoder_levels):
            xavier_init(
                self.input_convs[i].conv,
                gain=1,
                bias=0,
                distribution='uniform')
            
        trunc_normal_(self.lang_in_linear.weight, std=.02)
        nn.init.constant_(self.lang_in_linear.bias, 0)

        for i in range(0, self.num_input_levels - self.num_encoder_levels):
            caffe2_xavier_init(self.lateral_convs[i].conv, bias=0)
            caffe2_xavier_init(self.output_convs[i].conv, bias=0)

        self.mask_feature.apply(caffe2_xavier_init)
        self.lang_embed.apply(caffe2_xavier_init)

    def forward(self, feats, lang, lang_mask):
        """
        Args:
            feats (list[Tensor]): Feature maps of each level. Each has
                shape of (batch_size, c_i, h, w).
            lang (Tensor): shape (batch_size, c_l, L).
            lang_mask (Tensor): shape (batch_size, L).

        Returns:
            tuple: A tuple containing the following:

            - mask_feature (Tensor): shape (batch_size, c, h, w).
            - multi_scale_features (list[Tensor]): Multi scale \
                    features, each in shape (batch_size, c, h, w).
        """
        # generate padding mask for each level, for each image
        batch_size = feats[0].shape[0]
        encoder_input_list = []
        padding_mask_list = []
        level_positional_encoding_list = []
        spatial_shapes = []
        reference_points_list = []
        for i in range(self.num_encoder_levels):
            level_idx = self.num_input_levels - i - 1
            feat = feats[level_idx]
            feat_projected = self.input_convs[i](feat)
            h, w = feat.shape[-2:]

            # no padding
            padding_mask_resized = feat.new_zeros(
                (batch_size, ) + feat.shape[-2:], dtype=torch.bool)
            pos_embed = self.postional_encoding(padding_mask_resized)
            level_embed = self.level_encoding.weight[i]
            level_pos_embed = level_embed.view(1, -1, 1, 1) + pos_embed
            # (h_i * w_i, 2)
            reference_points = self.point_generator.single_level_grid_priors(
                feat.shape[-2:], level_idx, device=feat.device)
            # normalize
            factor = feat.new_tensor([[w, h]]) * self.strides[level_idx]
            reference_points = reference_points / factor

            # shape (batch_size, c, h_i, w_i) -> (batch_size, h_i * w_i, c)
            feat_projected = feat_projected.flatten(2).permute(0, 2, 1)
            level_pos_embed = level_pos_embed.flatten(2).permute(0, 2, 1)
            padding_mask_resized = padding_mask_resized.flatten(1)

            encoder_input_list.append(feat_projected)
            padding_mask_list.append(padding_mask_resized)
            level_positional_encoding_list.append(level_pos_embed)
            spatial_shapes.append(feat.shape[-2:])
            reference_points_list.append(reference_points)

        # shape (batch_size, total_num_query), total_num_query=sum([., h_i * w_i,.])
        padding_masks = torch.cat(padding_mask_list, dim=1)
        # shape (batch_size, total_num_query, c)
        encoder_inputs = torch.cat(encoder_input_list, dim=1)
        level_positional_encodings = torch.cat(level_positional_encoding_list, dim=1)
        device = encoder_inputs.device
        # shape (num_encoder_levels, 2), from low
        # resolution to high resolution
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=device)
        # shape (0, h_0*w_0, h_0*w_0+h_1*w_1, ...)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        reference_points = torch.cat(reference_points_list, dim=0)
        reference_points = reference_points[None, :, None].repeat(
            batch_size, 1, self.num_encoder_levels, 1)
        valid_radios = reference_points.new_ones(
            (batch_size, self.num_encoder_levels, 2))
        
        if self.with_prompts:
            prompts = self.vis_prompts.weight.unsqueeze(0).expand(batch_size, -1, -1) # [B, P, C]
            extend_l_mask = torch.cat([lang_mask, torch.ones(batch_size, self.num_prompts).to(lang_mask)], dim=1) # mask=0 to be ignored
        else:
            extend_l_mask = lang_mask
            prompts = None
        # shape (batch_size, num_total_query, c)
        memory = encoder_inputs
        lang = self.lang_in_norm(self.lang_in_linear(lang)) # [B, N_l, C]
        for i in range(self.num_enc_layers):
            memory, lang, prompts = self.encoder[i](
                vis=memory,
                lang=lang,
                lang_mask=extend_l_mask,
                prompts=prompts,
                vis_pos=level_positional_encodings,
                vis_padding_mask=padding_masks,
                spatial_shapes=spatial_shapes,
                reference_points=reference_points,
                level_start_index=level_start_index,
                valid_radios=valid_radios)
        # (batch_size, num_total_query, c) -> (batch_size, c, num_total_query)
        memory = memory.transpose(1, 2)

        # from low resolution to high resolution
        num_query_per_level = [e[0] * e[1] for e in spatial_shapes]
        outs = torch.split(memory, num_query_per_level, dim=-1)
        outs = [
            x.reshape(batch_size, -1, spatial_shapes[i][0],
                      spatial_shapes[i][1]) for i, x in enumerate(outs)
        ]

        for i in range(self.num_input_levels - self.num_encoder_levels - 1, -1,
                       -1):
            x = feats[i]
            cur_feat = self.lateral_convs[i](x)
            y = cur_feat + F.interpolate(
                outs[-1],
                size=cur_feat.shape[-2:],
                mode='bilinear',
                align_corners=False)
            y = self.output_convs[i](y)
            outs.append(y)

        mask_feature = self.mask_feature(outs[-1]) # [B, C, H, W]
        lang_g = self.lang_embed(lang[:, 0].unsqueeze(1)) # [B, 1, C], [CLS] embeddings
        prompts_g = self.lang_embed(torch.mean(prompts, dim=1, keepdim=True))
        lang_cls = torch.cat([prompts_g, lang_g], dim=1) # [B, 2, C]
        mask_pred = torch.einsum('bqc,bchw->bqhw', lang_cls, mask_feature) # [B, 2, H, W]

        return mask_pred