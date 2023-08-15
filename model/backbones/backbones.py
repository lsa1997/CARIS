import torch
from torch import nn
from torch.nn import functional as F
from .bert_encoder import MultiStageBertModel
from .swin_encoder import MultiStageSwinTransformer
from timm.models.layers import trunc_normal_

class PromptEncoder(nn.Module):
    def __init__(self, args, channels, bert_out_layers=[3, 6, 9, 12], prompt_levels=[1, 2], **kwargs):
        super().__init__()
        assert len(bert_out_layers) == 4 and len(channels) == 4, 'Only 4-stage index is supported!'
        self.bert_out_layers = bert_out_layers
        self.n_stages = 4
        self.text_dim = 768
        self.vis_dim = max(channels)
        self.vis_encoder = MultiStageSwinTransformer(**kwargs)
        self.lang_encoder = MultiStageBertModel.from_pretrained(args.ck_bert)
        self.lang_encoder.pooler = None
        self.conditional_prompts = ConditionalPrompt(self.vis_dim, self.text_dim, prompt_levels)
        self.num_prompts = self.conditional_prompts.num_prompts

    def forward(self, x, text, l_mask):
        '''
            Args:
                x: [B, C, H, W]
                text: [B, N_l]
                l_mask:[B, N_l]
            Returns:
                vis_outs (list): multi-level visual features
                l_feats: [B, N_l, 768]
        '''
        # Vis encoding
        vis_outs = []
        v_feats, Wh, Ww = self.vis_encoder.forward_embeddings(x)
        for stage in range(self.n_stages):
            v_feats, Wh, Ww = self.vis_encoder.forward_stages(v_feats, Wh, Ww, stage)
            vis_outs.append(self.vis_encoder.forward_norms(v_feats, Wh, Ww, stage)) # collect visual features before fusion
            v_feats, Wh, Ww = self.vis_encoder.forward_downs(v_feats, Wh, Ww, stage) # downsample
        # Text encoding
        prompts, l_mask = self.conditional_prompts(vis_outs[-1], l_mask)
        l_feats, extended_l_mask = self.lang_encoder.forward_embeddings(text, attention_mask=l_mask)
        l_feats = torch.cat([l_feats, prompts], dim=1) # [B, N_l+num_prompts, 768]
        l_feats = self.lang_encoder.forward_stages(l_feats, 0, 12, extended_l_mask) # [B, N_l+num_prompts, 768]
        return vis_outs, l_feats, l_mask

class ConditionalPrompt(nn.Module):
    def __init__(self, vis_dim, text_dim, prompt_levels=[1, 2]):
        super().__init__()
        self.num_levels = len(prompt_levels)
        self.pools = nn.ModuleList()
        self.in_linears = nn.ModuleList()
        self.out_linears = nn.ModuleList()
        num_prompts = 0
        for level in prompt_levels:
            num_prompts += level * level
            pool = nn.AdaptiveAvgPool2d((level, level))
            in_linear = nn.Linear(vis_dim, text_dim)
            out_linear = nn.Linear(text_dim, text_dim)
            self.pools.append(pool)
            self.in_linears.append(in_linear)
            self.out_linears.append(out_linear)
        self.num_prompts = num_prompts
        self.lang_prompts = nn.Embedding(self.num_prompts, text_dim)
        self.lang_prompts_norm = nn.LayerNorm(text_dim)
        self.init_weights()

    def init_weights(self):
        for i in range(self.num_levels):
            trunc_normal_(self.in_linears[i].weight, std=.02)
            nn.init.constant_(self.in_linears[i].bias, 0)
            trunc_normal_(self.out_linears[i].weight, std=.02)
            nn.init.constant_(self.out_linears[i].bias, 0)

    def forward(self, vis, l_mask):
        B = vis.shape[0]
        pool_vis_list = []
        for i in range(self.num_levels):
            pool_vis = self.pools[i](vis).flatten(2).transpose(1,2) # [B, n, vis_dim]
            pool_vis = self.out_linears[i](F.relu(self.in_linears[i](pool_vis))) # [B, n, text_dim]
            pool_vis_list.append(pool_vis)

        vis_prompts = torch.cat(pool_vis_list, dim=1) # [B, N, text_dim]
        lang_prompts = self.lang_prompts.weight.unsqueeze(0) # [1, num_prompts, text_dim]
        conditional_prompts = lang_prompts + vis_prompts # [B, num_prompts, text_dim]
        
        conditional_prompts = self.lang_prompts_norm(conditional_prompts)
        l_mask = torch.cat([l_mask, torch.ones(B, self.num_prompts).to(l_mask)], dim=1) # [B, N_l+num_prompts]
        return conditional_prompts, l_mask