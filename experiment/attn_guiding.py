import torch
from torch import nn, Tensor
import torch.nn.functional as F
import pdb

class AttnGuideReg(nn.Module):
    def __init__(self, num_heads, batch_size, loss_type, reduce=True, reduction='mean', *args, **kwargs):
        super(AttnGuideReg, self).__init__()
        self.num_heads = num_heads
        self.batch_size = batch_size
        self.reduce = reduce
        self.reduction = reduction
        self.loss_type = loss_type
    
    def gather_attn_map(self, attn_wts):
        '''
        Gather attention map for loss calculation
        '''
        attn_wts_gathered = []
        for s in range(len(attn_wts)):
            P = attn_wts[s][0].shape[-1]
            attn_map = []
            for l in range(len(attn_wts[s])):
                attn = attn_wts[s][l].view(self.batch_size, -1, P, P)[:,:self.num_heads,:,:] # B x n_h x P x P
                attn = attn.mean(dim=-1) # B x n_h x P 
                attn_map.append(attn)
            attn_wts_gathered.append(attn_map)
        return attn_wts_gathered

    def compute_frobenius_norm_loss(self, attn_wts, target):
        '''
        Compute frobenius norm loss for attention map
        :params attn_wts: [[[attn_map_layer1], [attn_map_layer2], ...], ...] each attn map is [B x n_h x P x P], len=num_scales
        :params target: [[B x sqrt(p) x sqrt(p)], ...] len=num_scales
        :return loss
        '''
        F_norm = torch.zeros(self.batch_size).to(target[0].device)
        for s in range(len(attn_wts)):
            for l in range(len(attn_wts[s])):
                F_norm += ((attn_wts[s][l] - target[s].view(self.batch_size,1,-1)) ** 2).sum(dim=(1,2))

        if self.reduce:
            F_norm = F_norm.sum()
        return F_norm
    
    def compute_inclusion_exclusion_loss(self, attn_wts, target):
        '''
        Compute inclusion & exclusion loss for attention map
        :params attn_wts: [[[attn_map_layer1], [attn_map_layer2], ...], ...] each attn map is [B x n_h x P x P], len=num_scales
        :params target: [[B x sqrt(p) x sqrt(p)], ...] len=num_scales
        :return loss
        '''
        loss = torch.zeros(self.batch_size).to(target[0].device)
        for s in range(len(attn_wts)):
            for l in range(len(attn_wts[s])):
                loss += -(attn_wts[s][l] * (target[s].view(self.batch_size,1,-1) > 0)).sum(dim=(1,2)) * 1e-2
                loss += (attn_wts[s][l] * (target[s].view(self.batch_size,1,-1) == 0)).sum(dim=(1,2)) * 1e-2

        if self.reduce:
            loss = loss.sum()
        return loss
    
    def compute_loss(self, pred, target):
        if self.loss_type == "Frobenius":
            return self.compute_frobenius_norm_loss(pred, target)
        if self.loss_type == "Inclusion_Exclusion":
            return self.compute_inclusion_exclusion_loss(pred, target)
        raise ValueError('Unknown loss type')

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        pred = self.gather_attn_map(pred)
        loss = self.compute_loss(pred, target)
        if self.reduction == 'mean':
            loss /= self.batch_size
        return loss