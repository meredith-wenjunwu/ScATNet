import torch
import torch.nn as nn
import pdb
import math



class AttentionLayer(nn.Module):
    def __init__(self, in_channels, n_heads=4, p=0.2):
        super(AttentionLayer, self).__init__()
        assert in_channels % n_heads == 0
        head_dim = in_channels // n_heads
        self.qkv = nn.Linear(in_channels, in_channels * 3)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(p=p)
        self.projection = nn.Linear(in_channels, in_channels)

        self.head_dim = head_dim
        self.n_heads = n_heads

    def forward(self, data):
        # [B X C X 1280]
        x = data['image']
        mask_ind = data['mask']
        assert x.dim() == 3
        B, C, D = x.size()

        qkv = self.qkv(x)
        q, k, v = torch.chunk(qkv, chunks=3, dim=-1)

        # B x C x D --> B x C x h x d --> B x h x C x d
        q = q.contiguous().view(B, C, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.contiguous().view(B, C, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.contiguous().view(B, C, self.n_heads, self.head_dim).transpose(1, 2)

        # [B x h x C x d] x [B x h x d x C] --> [B x h x C x C]
        attn = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if mask_ind is not None:
            mask = torch.ones((B, C, C)).cuda()

            for batch in range(B):
                mask_ind_batch = mask_ind[batch][mask_ind[batch] != -1].long()
                mask[batch, :, mask_ind_batch] = 0
            attn = attn.masked_fill(
                mask.unsqueeze(1).to(torch.bool), float('-inf')
            )
        #print(torch.max(attn))
        #print(torch.min(attn))
        attn = self.softmax(attn)
        # apply mask on row
        if mask_ind is not None:
            mask = torch.ones((B, C, C)).cuda()
            for batch in range(B):
                mask_ind_batch = mask_ind[batch][mask_ind[batch] != -1].long()
                mask[batch, mask_ind_batch, :] = 0

            attn = attn.masked_fill(
                mask.unsqueeze(1).to(torch.bool), float(0)
            )

        attn = self.dropout(attn)

        # [B x h x C x C] x [B x h x C x d] --> [B x h X C x d]
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2)
        out = out.contiguous().view(B, C, -1)
        return {'image': self.projection(out), 'mask': mask_ind}

class TransformerLayer(nn.Module):
    def __init__(self, in_channels, n_heads=4, p=0.2, linear_channel=4):
        super().__init__()
        self.attn = nn.Sequential(
            AttentionLayer(in_channels=in_channels, n_heads=n_heads),
        )
        self.dropout = nn.Dropout(p=p)

        self.ffn = nn.Sequential(
            nn.Linear(in_channels, linear_channel*in_channels), # maybe reduce 4 --> 2
            nn.ReLU(inplace=True),
            nn.Dropout(p=p), # maybe use higher dropout value
            nn.Linear(in_channels * linear_channel, in_channels),
            nn.Dropout(p=p)
        )
        self.norm_1 = nn.LayerNorm(in_channels)
        self.norm_2 = nn.LayerNorm(in_channels)

    def forward(self, data):
        x = data['image']
        mask = data['mask']
        res = x

        x = self.norm_1(x)

        data = self.attn({'image': x, 'mask':mask})
        x = data['image']
        mask = data['mask']
        x = self.dropout(x)
        x = x + res

        res = x
        x = self.norm_2(x)
        x = self.ffn(x)
        return {'image': x + res, 'mask': mask}

class TransformerLayer_scale(nn.Module):
    def __init__(self, in_channels, n_heads=4, p=0.2, linear_channel=4):
        super().__init__()
        self.attn = nn.Sequential(
            AttentionLayer(in_channels=in_channels, n_heads=n_heads),
        )
        self.dropout = nn.Dropout(p=p)

        self.ffn = nn.Sequential(
            nn.Linear(in_channels, linear_channel*in_channels), # maybe reduce 4 --> 2
            nn.ReLU(inplace=True),
            nn.Dropout(p=p), # maybe use higher dropout value
            nn.Linear(in_channels * linear_channel, in_channels),
            nn.Dropout(p=p)
        )
        self.norm_1 = nn.LayerNorm(in_channels)
        self.norm_2 = nn.LayerNorm(in_channels)

    def forward(self, data):
        x = data['image']
        mask = data['mask']
        res = x

        x = self.norm_1(x)

        data = self.attn({'image': x, 'mask':mask})
        x = data['image']
        mask = data['mask']
        x = self.dropout(x)
        x = x + res

        res = x
        x = self.norm_2(x)
        x = self.ffn(x)
        return {'image': x + res, 'mask': mask}


class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, height, width):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._positionalencoding2d(d_hid, height, width))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''

        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def _positionalencoding2d(self, d_model, height, width):
        """
        :param d_model: dimension of the model
        :param height: height of the positions
        :param width: width of the positions
        :return: d_model*height*width position matrix
        """
        if d_model % 4 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dimension (got dim={:d})".format(d_model))
        pe = torch.zeros(d_model, height, width)
        # Each dimension use half of d_model
        d_model = int(d_model / 2)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pos_w = torch.arange(0., width).unsqueeze(1)
        pos_h = torch.arange(0., height).unsqueeze(1)
        pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
        pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
        # [feat_dim x H x W] --? [H x W x feat_dim]
        pe = pe.permute(1, 2, 0)
        height, width, d_model = pe.size()
        pe = torch.reshape(pe, (height * width, d_model))
        return pe

    def forward(self, x):
        bsz = x.size(0)
        for i in range(bsz):
            pos = self.pos_table[:x.size(1), :].clone().detach().cuda()
            x[i] = x[i] + pos
        return x


class CNN(nn.Module):


    def __init__(self, configs):
        super(CNN, self).__init__()
        self.avgPool = nn.AdaptiveAvgPool1d(1)

        model_dim = configs['model_dim']  # potential: 512
        n_layers = configs['n_layers'] # potential: 2, 4, 6, 8

        self.mask_type=configs['mask_type']
        head_dim = configs['head_dim']
        num_scale = configs['multi_scale']

        assert model_dim % head_dim  == 0
        n_heads = model_dim // head_dim
        self.proj_layer = nn.Linear(1280, model_dim)    # add variable

        modules_attn = []
        for j in range(num_scale):
            # modules_attn_s = []
            # for i in range(n_layers):
                # layer = TransformerLayer(n_heads=n_heads, in_channels=model_dim,
                #                          p=configs['drop_out'],
                #                          linear_channel=configs['linear_channel'])
            layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=n_heads,
                                               dim_feedforward=configs['linear_channel'] * model_dim,
                                               dropout=configs['drop_out'])
            modules_attn_s = nn.TransformerEncoder(layer, num_layers=n_layers,
                                                   norm=nn.LayerNorm(model_dim))
                #modules_attn_s.append(layer)
            modules_attn.append(modules_attn_s)
        # self.attn_layer = [nn.Sequential(*modules_attn_s) for modules_attn_s in modules_attn]
        self.attn_layer = nn.ModuleList(modules_attn)
        # self.scale_attn_layer = TransformerLayer_scale(n_heads=n_heads, in_channels=model_dim,
                                                       # p=configs['drop_out'],
                                                       # linear_channel=configs['linear_channel'])
        scale_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=n_heads,
                                                 dim_feedforward=configs['linear_channel'] * model_dim,
                                                 dropout=configs['drop_out'])
        self.scale_attn_layer = nn.TransformerEncoder(scale_layer, num_layers=1, norm=nn.LayerNorm(model_dim))

        self.classifier = nn.Sequential(
            nn.Linear(model_dim, 64),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Linear(64, configs['num_classes'])
        )

    def forward(self, x):
        multi_data = x['image']
        multi_sizes = x['sizes']
        if self.mask_type != 'return-indices':
            multi_mask = [None] * len(multi_data)
        else:
            multi_mask = x['mask']

        multi_scale_feat = None
        for i in range(len(multi_data)):
            data = multi_data[i]
            mask = multi_mask[i]
            sizes = multi_sizes[i]
            bsz, n_crops, feat_dim = data.size()

            '''
            Create position encoding, assume that the sizes 
            are the same in one batch
            '''
            h, w = sizes[0].tolist()
            # position_enc = PositionalEncoding(feat_dim, h, w)
            # enc_output = position_enc(data)
            data = self.proj_layer(data)
            #multi_data[i] = data
            # reshape input to [HxW x B x E]
            # mask: [HxW  x HxW]
            B, C, _ = data.size()
            data = data.permute(1, 0, 2)
            mask_new = torch.ones((B, C)).cuda()

            for batch in range(B):
                mask_ind_batch = mask[batch][mask[batch] != -1].long()
                mask_new[batch, mask_ind_batch] = 0
            mask_new = mask_new.to(torch.bool)
            data = self.attn_layer[i](data, src_key_padding_mask=mask_new)
            out = data.permute(1, 0, 2)
            #mask = data['mask']
            out_vis = torch.sum(out ** 2, dim=-1, keepdim=True) / out.size(-1)
            out_vis = out_vis ** 0.5
            out = torch.mean(out, dim=1)
            if multi_scale_feat is None:
                multi_scale_feat = out.unsqueeze(0)
            else:
                multi_scale_feat = torch.cat([multi_scale_feat, out.unsqueeze(0)], dim=0)
        multi_scale_feat = multi_scale_feat.transpose(0, 1)
        data = self.scale_attn_layer(multi_scale_feat)
        out = data
        out = torch.mean(out, dim=1)
        out = self.classifier(out)
        return out, out_vis


if __name__ == '__main__':
    import pdb
    import argparse
    parser = argparse.ArgumentParser()

    configs = parser.parse_args()
    configs.num_classes = 4
    configs.model_dim = 512
    configs.n_layers = 4
    configs.head_dim = 64
    configs.mask_type=None
    configs.drop_out = 0.4
    configs.linear_channel = 4

    x = dict()
    x['image'] = torch.rand(2, 400, 1280)
    x['mask'] = None
    x['sizes'] = [torch.tensor([16, 25])]
    layer = CNN(configs=vars(configs))
    out,_ = layer(x)
    print(out.shape)
