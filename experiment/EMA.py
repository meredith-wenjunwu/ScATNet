from copy import deepcopy
import torch

class EMA(object):
    def __init__(self, model, ema_momentum: float = 0.1, device: str = ''):
        # make a deep copy of the model for accumulating moving average of parameters
        self.ema_model = deepcopy(model)
        self.ema_model.eval()
        self.momentum = ema_momentum
        self.device = device
        if device:
            self.ema_model.to(device=device)
        self.ema_has_module = hasattr(self.ema_model, 'module')
        for param in self.ema_model.parameters():
            param.requires_grad = False


    def update_parameters(self, model):
        # correct a mismatch in state dict keys
        has_module = hasattr(model, 'module') and not self.ema_has_module
        with torch.no_grad():
            msd = model.state_dict()
            for k, ema_v in self.ema_model.state_dict().items():
                if has_module:
                    k = 'module.' + k
                model_v = msd[k].detach()
                if self.device:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_((ema_v * (1.0 - self.momentum)) + (self.momentum * model_v))