import torch
import torch.nn as nn
from torch.nn import init
import pdb


from typing import List, Optional

class MeanWeight(nn.Module):
    def __init__(self, configs):
        super(MeanWeight, self).__init__()

        self.classifier = nn.Sequential(
            nn.Linear(1280, 256),
            nn.SiLU(),
            nn.Linear(256, configs['num_classes'])
        )

    def forward(self, x):
        x = torch.mean(x, dim=1)
        out = self.classifier(x)
        return out, 1



class LearnedWeight(nn.Module):
    def __init__(self, configs):
        super(LearnedWeight, self).__init__()
        self.num_of_crops = int((configs['resize1'][0][0]/configs['resize2']) * (configs['resize1'][0][1]/configs['resize2']))
        self.num_scales = configs['multi_scale']
        if self.num_scales > 1:
            self.mean_weights = nn.Parameter(torch.Tensor(size=(self.num_scales,)).fill_(1.0/self.num_scales))
        else:
            self.mean_weights = nn.Parameter(torch.Tensor(size=(self.num_of_crops,)).fill_(1.0 / self.num_of_crops))
        self.classifier = nn.Sequential(
            nn.Linear(1280, 256),
            nn.SiLU(),
            nn.Linear(256, configs['num_classes'])
        )

    def forward(self, x: List[torch.Tensor], src_mask: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        if self.num_scales > 1:
            scale_embeds = []
            for x_s in x:
                scale_embeds.append(torch.mean(x_s, dim=1))
            scale_embeds = torch.stack(scale_embeds, dim=-1)
            scale_embeds = torch.matmul(scale_embeds, self.weighted_avg)
            out = self.classifier(scale_embeds)
        else:
            x = x[0]
            x = torch.matmul(self.mean_weights, x)
            out = self.classifier(x)
        return out

# TODO: try this one msc_model.py
class SelfAttn(nn.Module):
    def __init__(self, configs):
        super(SelfAttn, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.classifier = nn.Sequential(
            nn.Linear(1280, 256),
            nn.SiLU(),
            nn.Linear(256, configs['num_classes'])
        )

    def forward(self, x):
        attn = torch.mean(x, dim=-1)
        # [B x C] --> [B x C x 1]
        attn = self.softmax(attn).unsqueeze(2)
        # [B x feat_dim x C] x [B x C x 1] --> [B x feat_dim x 1]
        out = torch.matmul(x.transpose(1,2), attn).squeeze(2)
        out = self.classifier(out)
        return out



