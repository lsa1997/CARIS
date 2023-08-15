import torch
from torch import nn
from torch.nn import functional as F
from bert.modeling_bert import BertModel

class MultiStageBertModel(BertModel):
    '''
        BertModel that supports multi-stage encoding for multimodal learning.
    '''
    def __init__(self, config):
        super().__init__(config)

    def forward_embeddings(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        return_attn_mask=True
    ):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device
        embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )

        if return_attn_mask:
            if attention_mask is None:
                attention_mask = torch.ones(input_shape, device=device)
            if token_type_ids is None:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

            # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
            # ourselves in which case we just need to make it broadcastable to all heads.
            extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, device)
            return embedding_output, extended_attention_mask
        else:
            return embedding_output

    def forward_stages(
        self,
        hidden_states,
        start_layer,
        end_layer,
        attention_mask=None,
    ):
        assert end_layer > start_layer, 'end_layer {} is not larger than start layer {}.'.format(end_layer, start_layer)

        for i in range(start_layer, end_layer):
            layer_outputs = self.encoder.layer[i](
                hidden_states,
                attention_mask=attention_mask,
            )
            hidden_states = layer_outputs[0]

        return hidden_states # (B, N_l, 768)
