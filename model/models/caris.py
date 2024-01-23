import torch
from torch import nn
from torch.nn import functional as F

class CARIS(nn.Module):
    def __init__(self, backbone, pixel_decoder, args, num_classes=1, criterion=None):
        super(CARIS, self).__init__()
        self.backbone = backbone
        self.pixel_decoder = pixel_decoder
        self.num_classes = num_classes

        self.criterion = criterion
        self.base_lr = args.lr

    def params_to_optimize(self, scale_lang=0.1, scale_vis=0.1):
        # parameters to optimize
        names_frozen = list()
        names_no_decay = list()
        lang_backbone_names_no_decay = list()
        lang_backbone_params_no_decay = list()
        lang_backbone_params_decay = list()
        backbone_names_no_decay = list()
        backbone_params_no_decay = list()
        backbone_params_decay = list()
        params_no_decay = list()
        params_decay = list()
        for name, m in self.named_parameters():
            if m.requires_grad:
                if 'backbone' in name:
                    # Language backbone
                    if 'lang_encoder' in name:
                        if 'Norm' in name:
                            lang_backbone_params_no_decay.append(m)
                            lang_backbone_names_no_decay.append(name)
                        elif 'embeddings' in name:
                            lang_backbone_params_no_decay.append(m)
                            lang_backbone_names_no_decay.append(name)
                        else:
                            lang_backbone_params_decay.append(m)
                    # Visual backbone
                    elif 'vis_encoder' in name:
                        if 'norm' in name:
                            backbone_params_no_decay.append(m)
                            backbone_names_no_decay.append(name)
                        elif 'absolute_pos_embed' in name or 'relative_position_bias_table' in name:
                            backbone_params_no_decay.append(m)
                            backbone_names_no_decay.append(name)
                        elif 'position_embeddings' in name:
                            backbone_params_no_decay.append(m)
                            backbone_names_no_decay.append(name)
                        else:
                            backbone_params_decay.append(m)
                    # Others
                    elif 'lang_prompts' in name:
                        params_no_decay.append(m)
                        names_no_decay.append(name)
                    elif 'norm' in name:
                        params_no_decay.append(m)
                        names_no_decay.append(name)
                    else:
                        params_decay.append(m)
                else:
                    if 'norm' in name or 'Norm' in name:
                        params_no_decay.append(m)
                        names_no_decay.append(name)
                    elif 'absolute_pos_embed' in name or 'relative_position_bias_table' in name:
                        params_no_decay.append(m)
                        names_no_decay.append(name)
                    elif 'prompt' in name:
                        params_no_decay.append(m)
                        names_no_decay.append(name)
                    else:
                        params_decay.append(m)
            else:
                names_frozen.append(name)

        params_to_optimize = [
            {'params': lang_backbone_params_no_decay, 'weight_decay': 0.0, 'lr': scale_lang * self.base_lr},
            {'params': lang_backbone_params_decay, 'lr': scale_lang * self.base_lr},
            {'params': backbone_params_no_decay, 'weight_decay': 0.0, 'lr': scale_vis * self.base_lr},
            {'params': backbone_params_decay, 'lr': scale_vis * self.base_lr},
            {'params': params_no_decay, 'weight_decay': 0.0, 'lr': self.base_lr},
            {'params': params_decay, 'lr': self.base_lr},
        ]
        print('scale_lang_backbone: ', scale_lang)
        print('scale_vis_backbone: ', scale_vis)
        print('LANG BACKBONE NO DECAY params: ', lang_backbone_names_no_decay)
        print('BACKBONE NO DECAY params: ', backbone_names_no_decay)
        print('NO DECAY params: ', names_no_decay)
        print('FROZEN params: ', names_frozen)
        return params_to_optimize

    def forward(self, x, text, l_mask, resize_output=True, targets=None, return_probs=False, return_attn=False):
        '''
            Input:
                x       [BxCxHxW]
                text    [BxN_l]
                l_mask  [BxN_l]
        '''
        input_shape = x.shape[-2:]
        lang_len = l_mask.shape[1]

        # Multi-modal encoding
        outs = self.backbone(x, text, l_mask) #vis_outs[-1]: [B, C, H, W] l_feats: [B, N_l, 768]
        vis_outs = outs[0]
        l_feats = outs[1]
        # VL pixel decoder
        l_feats = l_feats[:,:lang_len] # [B, N_l, 768]
        if return_attn:
            x, attns = self.pixel_decoder(vis_outs, l_feats, l_mask, return_attn=return_attn) # [B, 1, H, W]
        else:
            x = self.pixel_decoder(vis_outs, l_feats, l_mask) # [B, 1, H, W]

        if self.training:
            if self.criterion is not None:
                losses = self.criterion(x, targets)
                return losses

        if resize_output:
            x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=True)
            if return_attn:
                attns = [F.interpolate(attn, size=input_shape, mode='bilinear', align_corners=True) for attn in attns]
                attns = [attn.reshape(x.shape[0], self.pixel_decoder.num_enc_layers, -1, input_shape[0], input_shape[1]) for attn in attns] # [B, N_layer, N_l, H, W]
        if x.shape[1] == 1:
            if not return_probs:
                x = x.sigmoid()
                x = (x >= 0.5) * 1
        else:
            if not return_probs:
                x = torch.argmax(x, dim=1, keepdim=True)
        if return_attn:
            return x, attns
        return x
