import torch
import torch.nn as nn
from torch.nn import init

class CNN(nn.Module):


    def __init__(self, configs):
        super(CNN, self).__init__()
        self.avgPool = nn.AdaptiveAvgPool1d(1)
        self.chunks = 3
        self.proj_layer = nn.Linear(1280, 256*self.chunks)
        self.softmax = nn.Softmax(dim=-1)

        self.proj_after_attn = nn.Sequential(
            nn.Linear(256, 64),
            nn.LeakyReLU(inplace=True, negative_slope=0.1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Linear(32, configs['num_classes'])
        )

    def forward(self, x):
        data = x['image']
        # data is [B x C x 1280] --> [B x C x 256]
        data = self.proj_layer(data)

        q, k , v = torch.chunk(data, chunks=self.chunks, dim=-1)

        # [B x C x 256] x [B x 256 x C] --> [B x C x C]
        attn_matrix = torch.bmm(q, k.transpose(1, 2))
        # [B x C x C] --> softmax on last dimension
        attn_matrix = self.softmax(attn_matrix).float()

        # [B x C x C] x [B x C x 256] --> [B x C x 256]
        out = torch.bmm(attn_matrix, v)
        out_vis = torch.sum(out ** 2, dim=-1, keepdim=True) / out.size(-1)
        out_vis = out_vis ** 0.5

        # [B x C x 256] --> [B x C x 64]
        out = self.proj_after_attn(out)

        # [B x C x 64] --> [B x 64]
        out = torch.mean(out, dim=1)

        out = self.classifier(out)
        return out, out_vis


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    configs = parser.parse_args()
    configs.num_classes = 4

    x = dict()
    x['image'] = torch.Tensor(3, 5, 1280)
    layer = CNN(configs=configs)
    out = layer(x)
    print(out.size())
