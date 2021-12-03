import torch
from torch import nn, Tensor
import math
from typing import Optional, List
import pdb
from model.modules.positional_embedding import PositionalEncoding
from model.drop_layers import RecurrentDropout


class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim: int, n_heads: int, attn_drop: Optional[float] = 0.1):
        assert model_dim % n_heads == 0
        super(MultiHeadAttention, self).__init__()
        head_dim = model_dim // n_heads
        # multiplied by 3 because we have 3 branches (query, key , and value)
        self.kqv = nn.Linear(in_features=model_dim, out_features=model_dim * 3, bias=True)
        self.softmax = nn.Softmax(dim=-1)
        self.attn_dropout = nn.Dropout(p=attn_drop)
        self.out_proj = nn.Linear(in_features=model_dim, out_features=model_dim, bias=True)
        self.scaling = head_dim ** -0.5
        self.num_heads = n_heads
        self.head_dim = head_dim

    def forward(self, x, *args, **kwargs):
        # x -> [B x P x C]
        # [B x P x C] --> [P x B x C]
        x = x.transpose(0, 1)
        num_crops, batch_size, in_dim = x.shape
        qkv = self.kqv(x)
        q, k, v = torch.chunk(qkv, chunks=3, dim=-1)
        # [B x P x
        q = (
            q.contiguous().view(num_crops, batch_size * self.num_heads,
                                self.head_dim)  # [P x B x C] --> [P x B*n_h x h_dim]
                .transpose(0, 1)  # [P x B*n_h x h_dim] --> [B*n_h x P x h_dim]
        )
        k = (
            k.contiguous().view(num_crops, batch_size * self.num_heads,
                                self.head_dim)  # [P x B x C] --> [P x B*n_h x h_dim]
                .transpose(0, 1)  # [P x B*n_h x h_dim] --> [B*n_h x P x h_dim]
        )
        v = (
            # [P x B x C] --> [P x B*n_h x h_dim]
            v.contiguous().view(num_crops, batch_size * self.num_heads, self.head_dim)
                .transpose(0, 1)  # [P x B*n_h x h_dim] --> [B*n_h x P x h_dim]
        )
        # [B*n_h x P x h_dim] x [B*n_h x P x h_dim]^T --> [B*n_h x P x h_dim] x [B*n_h x h_dim x P] --> [B*n_h x P x P]
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        attn_weights = self.softmax(attn_weights)
        attn_weights = self.attn_dropout(attn_weights)
        out = torch.bmm(attn_weights, v)
        out = out.transpose(0, 1).contiguous().view(num_crops, batch_size, -1)
        out = self.out_proj(out).transpose(0, 1)
        if kwargs.get('need_attn_wts'):
            return out, attn_weights
        return out, None


class FFN(nn.Module):
    def __init__(self, model_dim, ffn_dim, act_fn: Optional[nn.Module] = nn.ReLU(), ffn_drop_p: Optional[float] = 0.1):
        super(FFN, self).__init__()
        self.ffn = nn.Sequential(
            nn.Linear(in_features=model_dim, out_features=ffn_dim, bias=True),
            act_fn,
            nn.Dropout(p=ffn_drop_p),
            nn.Linear(in_features=ffn_dim, out_features=model_dim, bias=True),
            nn.Dropout(p=ffn_drop_p)
        )

    def forward(self, x: Tensor) -> Tensor:
        # x -> [B x P x C]
        return self.ffn(x)


class TransformerLayer(nn.Module):
    def __init__(self, model_dim: int, n_heads: int, ffn_dim: int,
                 act_fn: Optional[nn.Module] = nn.ReLU(),
                 attn_drop: Optional[float] = 0.1, ffn_drop_p: Optional[float] = 0.1,
                 dropout_p: Optional[float] = 0.1
                 ):
        super(TransformerLayer, self).__init__()
        self.norm_before_attn = nn.LayerNorm(normalized_shape=model_dim)
        self.mha = MultiHeadAttention(model_dim=model_dim, n_heads=n_heads, attn_drop=attn_drop)
        self.norm_before_ffn = nn.LayerNorm(normalized_shape=model_dim)
        self.ffn = FFN(model_dim=model_dim, ffn_dim=ffn_dim, ffn_drop_p=ffn_drop_p, act_fn=act_fn)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x, *args, **kwargs):
        # x -> [B x P x C]
        # MHA part
        res = x
        x = self.norm_before_attn(x)
        x, attn_wts = self.mha(x, **kwargs)
        x = self.dropout(x)
        x = x + res
        # FFN Part
        res = x
        x = self.norm_before_ffn(x)
        x = self.ffn(x)
        x = x + res
        if attn_wts is not None:
            return x, attn_wts
        return x


class AttentionLayer(nn.Module):
    def __init__(self, in_dim: int, proj_dim: int, num_layers: int, ffn_dim: Optional[int] = None,
                 head_dim: Optional[int] = 64, act_fn: Optional[nn.Module] = nn.ReLU(),
                 attn_drop: Optional[float] = 0.1, dropout_p: Optional[float] = 0.1, variational_dropout=False):
        if proj_dim % head_dim != 0:
            proj_dim = int(math.ceil(proj_dim / head_dim) * head_dim)
        n_heads = proj_dim // head_dim
        if num_layers <= 0:
            num_layers = 1
        if ffn_dim is None:
            ffn_dim = 4 * proj_dim
        super(AttentionLayer, self).__init__()
        self.proj_cnn_emb = nn.Linear(in_features=in_dim, out_features=proj_dim, bias=True)
        # Patch positional embedding
        self.pos_emb = PositionalEncoding(d_model=proj_dim)
        self.emb_drop = RecurrentDropout(p=dropout_p, batch_first=True) if variational_dropout else nn.Dropout(
            p=dropout_p)
        self.attn = nn.Sequential()
        for i in range(num_layers):
            layer = TransformerLayer(model_dim=proj_dim, n_heads=n_heads, ffn_dim=ffn_dim,
                                     act_fn=act_fn, attn_drop=attn_drop, ffn_drop_p=dropout_p,
                                     dropout_p=dropout_p)
            self.attn.add_module(name="encoder_{}".format(i), module=layer)
        self.scaling = proj_dim ** 0.5
        self.proj_attn_emb = nn.Sequential(
            nn.LayerNorm(normalized_shape=proj_dim),
            nn.Linear(in_features=proj_dim, out_features=proj_dim, bias=True)
        )

    def forward(self, x: Tensor, scale_enc: Optional[Tensor] = None,
                src_mask: Optional[Tensor] = None) -> (Tensor, Tensor):
        # x -> [B x P x C]
        # apply mask on row and col to after line 152
        if src_mask is not None:
            x = x.masked_fill(src_mask.unsqueeze(2).to(torch.bool), 0)
        x = self.proj_cnn_emb(x) * self.scaling
        x = self.pos_emb(x) + scale_enc if scale_enc is not None else self.pos_emb(x)
        x = self.emb_drop(x)
        x = self.attn(x)
        out_vis = torch.sum(x ** 2, dim=-1, keepdim=True) / x.size(-1)
        out_vis = out_vis ** 0.5
        x = torch.mean(x, dim=1)
        x = self.proj_attn_emb(x)
        return x, out_vis


class MultiScaleAttention(nn.Module):
    def __init__(self, configs):
        super(MultiScaleAttention, self).__init__()
        self.mask_type = configs['mask_type']
        head_dim = configs['head_dim']
        self.multi_scale = configs['resize1_scale']
        in_dim = configs['in_dim']
        proj_dim = configs['model_dim']
        num_layers = configs['n_layers']
        num_classes = configs['num_classes']
        ffn_dim = configs['linear_channel'] * configs['model_dim']
        act_fn = nn.SiLU()
        attn_drop = configs['drop_out']
        dropout_p = configs['drop_out']
        assert len(self.multi_scale) >= 1, "Atleast one scale is required"
        self.scale_wise_attn = nn.Sequential()
        # scale embedding
        self.num_scales = len(self.multi_scale)
        weight_tying = configs['weight_tie']
        num_scale_attn = configs['num_scale_attn_layer']
        self.weight_tying = True if self.num_scales > 1 and weight_tying else False
        if not self.weight_tying:
            for i in range(self.num_scales):
                layer = AttentionLayer(in_dim=in_dim, proj_dim=proj_dim, num_layers=num_layers, ffn_dim=ffn_dim,
                                       head_dim=head_dim, act_fn=act_fn, attn_drop=attn_drop, dropout_p=dropout_p,
                                       variational_dropout=configs['variational_dropout'])
                self.scale_wise_attn.add_module(name="scale_{}".format(i), module=layer)
        else:
            layer = AttentionLayer(in_dim=in_dim, proj_dim=proj_dim, num_layers=num_layers, ffn_dim=ffn_dim,
                                   head_dim=head_dim, act_fn=act_fn, attn_drop=attn_drop, dropout_p=dropout_p,
                                   variational_dropout=configs['variational_dropout'])
            self.scale_wise_attn.add_module(name="scale_0", module=layer)
        self.attend_over_scales = None

        self.scale_pos_emb = None
        use_standard_emb = configs['use_standard_emb']
        if self.num_scales > 1:
            self.attend_over_scales = nn.Sequential()
            if use_standard_emb:
                self.scale_pos_emb = nn.Embedding(num_embeddings=num_classes, embedding_dim=proj_dim)
            else:
                self.scale_pos_emb = PositionalEncoding(d_model=proj_dim)
            self.scaling_factor = proj_dim ** 0.5
            for i in range(num_scale_attn):
                n_heads = proj_dim // head_dim
                layer = TransformerLayer(model_dim=proj_dim, n_heads=n_heads, ffn_dim=ffn_dim,
                                         act_fn=act_fn, attn_drop=attn_drop, ffn_drop_p=dropout_p,
                                         dropout_p=dropout_p)
                self.attend_over_scales.add_module(name="attn_over_scale_{}".format(i), module=layer)

            self.scale_fusion = nn.Sequential(
                nn.LayerNorm(normalized_shape=proj_dim * self.num_scales),
                nn.Linear(in_features=proj_dim * self.num_scales, out_features=proj_dim, bias=True),
                nn.LayerNorm(normalized_shape=proj_dim),
                act_fn,
            )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=proj_dim, out_features=64),
            act_fn,
            nn.Dropout(p=0.2),
            nn.Linear(in_features=64, out_features=num_classes, bias=True)
        )
        self.softmax = nn.Softmax(dim=-1)
        self.scale_emb_drop = RecurrentDropout(p=dropout_p, batch_first=True) \
            if configs['variational_dropout'] else nn.Dropout(p=0.15625)
        self.reset_params()

    def reset_params(self):
        '''
        Function to initialze the parameters
        '''
        from torch.nn import init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.LayerNorm):
                init.constant_(m.weight, 1.0)
                if m.bias is not None:
                    init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Embedding):
                embedding_dim = m.embedding_dim
                nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)

    def scale_encodings(self, batch_size, device):
        scale_encodings = []
        for scale_idx in range(self.num_scales):
            scale_encoding = torch.LongTensor(batch_size).fill_(scale_idx).to(device=device)
            scale_encoding = self.scale_pos_emb(scale_encoding)
            scale_encodings.append(scale_encoding)
        return torch.stack(scale_encodings, dim=1)  # [Batch x Scales x Feature_dim]

    def forward(self, x: List[Tensor], *args, **kwargs):
        '''
            input: A list containing tensors. Each tensor corresponds to embeddings for each scale and is of the shape
                ( batch_size, num_crops, cnn_dimension)
            output: A tensor of size (batch size x num_classes)
        '''
        assert len(x) == self.num_scales, "Expected {} scale, got {}".format(self.num_scales, len(x))
        scale_attn_vis = []
        attn = None
        scale_embeds = []
        bsz = x[0].size(0)
        device = x[0].device
        '''Added Index Scale Position Encoding'''
        if self.weight_tying:
            for x_s in x:
                # Batch x Crops x Feature
                out, out_vis = self.scale_wise_attn(x_s)
                scale_embeds.append(out)
                scale_attn_vis.append(out_vis)
            out = torch.stack(scale_embeds, dim=1)
        else:
            scale_embeds = []
            scale_idx = 0
            for x_s, layer in zip(x, self.scale_wise_attn):
                out, out_vis = layer(x=x_s)
                scale_embeds.append(out)
                scale_attn_vis.append(out_vis)
                scale_idx += 1
            out = torch.stack(scale_embeds, dim=1)

        if self.num_scales > 1:
            # [B x Scales x features]
            # scale_encodings = self.scale_encodings(bsz, device=device)
            '''normalization for scale embedding'''
            # out = self.scale_emb_norm(out)

            '''Position encoding for scale embedding'''
            if isinstance(self.scale_pos_emb, nn.Embedding):
                scale_encodings = self.scale_encodings(batch_size=bsz, device=device) * self.scaling_factor
                out = out + scale_encodings
            else:
                out = self.scale_pos_emb(out)

            '''drop out for scale position embedding'''
            out = self.scale_emb_drop(out)
            for l in self.attend_over_scales:
                out, attn = l(out, need_attn_wts=True)
            # out = torch.mean(out, dim=1)

            out = out.contiguous().view(out.size(0), -1)

            out = self.scale_fusion(out)
        else:
            out = out.contiguous().view(out.size(0), -1)
        out = self.classifier(out)
        return out, scale_attn_vis, attn  # .unsqueeze(0)


if __name__ == "__main__":
    configs = {}
    configs['mask_type'] = None
    configs['head_dim'] = 64
    configs['resize1_scale'] = [1.0]
    configs['model_dim'] = 256
    configs['in_dim'] = 1280
    configs['n_layers'] = 4
    configs['num_classes'] = 4
    configs['linear_channel'] = 4
    configs['drop_out'] = 0.4
    configs['self_attention'] = True
    configs['weight_tie'] = True
    configs['variational_dropout'] = False
    configs['num_scale_attn_layer'] = 2
    configs['use_standard_emb'] = 1

    layer = MultiScaleAttention(configs)

    # [B x C x F] x Scales
    inp_tensor = [torch.Tensor(1, 49, 1280)]
    out = layer(inp_tensor)
    print('final: {}'.format(out[0].shape))
