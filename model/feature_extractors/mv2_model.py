from torch import nn, Tensor
from typing import Optional
import torch
import pdb

def make_divisible(v, divisor=8, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class GlobalPool(nn.Module):
    def __init__(self, pool_type='mean', keep_dim=False):
        super(GlobalPool, self).__init__()
        self.pool_type = pool_type
        self.keep_dim = keep_dim

    def _global_pool(self, x):
        assert x.dim() == 4, "Got: {}".format(x.shape)
        x = torch.mean(x, dim=[-2, -1], keepdim=self.keep_dim)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._global_pool(x)


class LinearLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: Optional[bool] = True,
                 is_ws: Optional[bool] = False):
        super(LinearLayer, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        self.is_ws = is_ws
        self.in_features = in_features
        self.out_features = out_features

        self.momentum = 0.1
        self.one_minus_momentum = 1.0 - self.momentum
        if self.is_ws:
            self.register_buffer('running_mean', torch.zeros(out_features, 1))
            self.register_buffer('running_var', torch.ones(out_features, 1))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)

        self.reset_params()

    def reset_params(self):
        if self.weight is not None:
            torch.nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            torch.nn.init.constant_(self.bias, 0)

    def _standardize_wts(self, weights, eps=1e-9):
        if self.training:
            var, mean = torch.var_mean(weights, dim=1, keepdim=True, unbiased=False)
            with torch.no_grad():
                n = weights.numel() / self.out_features
                self.running_mean = (self.running_mean * self.one_minus_momentum) + (self.momentum * mean)
                self.running_var = (self.running_var * self.one_minus_momentum) + ((self.momentum * var * n) / (n - 1))
        else:
            var = self.running_var
            mean = self.running_mean

        std = torch.sqrt(var + eps)
        weights = torch.sub(weights, mean).div(std + eps)
        return nn.Parameter(weights)

    def forward(self, x: Tensor) -> Tensor:
        if self.is_ws:
            self.weight = self._standardize_wts(self.weight)
        if self.bias is not None and x.dim() == 2:
            x = torch.addmm(self.bias, x, self.weight.t())
        else:
            x = x.matmul(self.weight.t())
            if self.bias is not None:
                x += self.bias
        return x


class ConvLayer(nn.Module):
    def __init__(self, opts, in_channels: int, out_channels: int, kernel_size: int or tuple,
                 stride: Optional[int or tuple] = 1,
                 dilation: Optional[int or tuple] = 1, groups: Optional[int] = 1,
                 bias: Optional[bool] = False, padding_mode: Optional[str] = 'zeros',
                 use_norm: Optional[bool] = True, use_act: Optional[bool] = True,
                 padding: Optional[int or tuple] = (0, 0),
                 auto_padding: Optional[bool] = True):
        super(ConvLayer, self).__init__()

        if use_norm:
            assert not bias, 'Do not use bias when using normalization layers.'

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        if isinstance(stride, int):
            stride = (stride, stride)

        if isinstance(dilation, int):
            dilation = (dilation, dilation)

        assert isinstance(kernel_size, (tuple, list))
        assert isinstance(stride, (tuple, list))
        assert isinstance(dilation, (tuple, list))

        if auto_padding:
            padding = (int((kernel_size[0] - 1) / 2) * dilation[0], int((kernel_size[1] - 1) / 2) * dilation[1])

        block = nn.Sequential()
        conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias,
                               padding_mode=padding_mode)

        block.add_module(name="conv", module=conv_layer)

        self.norm_name = None
        if use_norm:
            norm_layer = nn.BatchNorm2d(num_features=out_channels)
            block.add_module(name="norm", module=norm_layer)
            self.norm_name = norm_layer.__class__.__name__

        self.act_name = None
        act_type = getattr(opts, "model.activation.name", "prelu")

        if act_type is not None and use_act:
            act_layer = nn.PReLU(num_parameters=out_channels)
            block.add_module(name="act", module=act_layer)
            self.act_name = act_layer.__class__.__name__

        self.block = block

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.groups = groups
        self.kernel_size = conv_layer.kernel_size
        self.bias = bias

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class InvertedResidual(nn.Module):
    def __init__(self, opts, in_channels: int, out_channels: int, stride: int, expand_ratio: int) -> None:
        assert stride in [1, 2]
        super(InvertedResidual, self).__init__()
        self.stride = stride

        hidden_dim = make_divisible(int(round(in_channels * expand_ratio)), 8)
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        block = nn.Sequential()
        if expand_ratio != 1:
            block.add_module(name="exp_1x1",
                             module=ConvLayer(opts, in_channels=in_channels, out_channels=hidden_dim, kernel_size=1,
                                              use_act=True, use_norm=True))

        block.add_module(name="conv_3x3",
                         module=ConvLayer(opts, in_channels=hidden_dim, out_channels=hidden_dim, stride=stride,
                                          kernel_size=3, groups=hidden_dim, use_act=True, use_norm=False))

        block.add_module(name="red_1x1",
                         module=ConvLayer(opts, in_channels=hidden_dim, out_channels=out_channels, kernel_size=1,
                                          use_act=False, use_norm=True))

        self.block = block
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.exp = expand_ratio

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        if self.use_res_connect:
            return x + self.block(x)
        else:
            return self.block(x)


class MobileNetV2(nn.Module):
    def __init__(self, opts):

        width_mult = getattr(opts, "model.classification.scale", 1.0)
        num_classes = getattr(opts, "model.classification.n_classes", 1000)
        mv2_config = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        image_channels = 3
        input_channels = 32
        last_channel = 1280
        super(MobileNetV2, self).__init__()
        self.round_nearest = 8

        last_channel = make_divisible(last_channel * max(1.0, width_mult), self.round_nearest)

        self.model_conf_dict = dict()

        self.conv_1 = ConvLayer(opts=opts, in_channels=image_channels, out_channels=input_channels,
                                kernel_size=3, stride=2, use_norm=True, use_act=True)
        self.model_conf_dict['conv1'] = {'in': image_channels, 'out': input_channels}

        self.layer_1, out_channels = self._make_layer(opts=opts,
                                                      mv2_config=mv2_config[0],
                                                      width_mult=width_mult,
                                                      input_channel=input_channels,
                                                      round_nearest=self.round_nearest)
        self.model_conf_dict['layer1'] = {'in': input_channels, 'out': out_channels}
        input_channels = out_channels

        self.layer_2, out_channels = self._make_layer(opts=opts,
                                                      mv2_config=mv2_config[1],
                                                      width_mult=width_mult,
                                                      input_channel=input_channels,
                                                      round_nearest=self.round_nearest)
        self.model_conf_dict['layer2'] = {'in': input_channels, 'out': out_channels}
        input_channels = out_channels

        self.layer_3, out_channels = self._make_layer(opts=opts,
                                                      mv2_config=mv2_config[2],
                                                      width_mult=width_mult,
                                                      input_channel=input_channels,
                                                      round_nearest=self.round_nearest)
        self.model_conf_dict['layer3'] = {'in': input_channels, 'out': out_channels}
        input_channels = out_channels

        self.layer_4, out_channels = self._make_layer(opts=opts,
                                                      mv2_config=[mv2_config[3], mv2_config[4]],
                                                      width_mult=width_mult,
                                                      input_channel=input_channels,
                                                      round_nearest=self.round_nearest)
        self.model_conf_dict['layer4'] = {'in': input_channels, 'out': out_channels}
        input_channels = out_channels

        self.layer_5, out_channels = self._make_layer(opts=opts,
                                                      mv2_config=[mv2_config[5], mv2_config[6]],
                                                      width_mult=width_mult,
                                                      input_channel=input_channels,
                                                      round_nearest=self.round_nearest)
        self.model_conf_dict['layer5'] = {'in': input_channels, 'out': out_channels}
        input_channels = out_channels

        self.conv_1x1_exp = ConvLayer(opts=opts, in_channels=input_channels, out_channels=last_channel,
                                      kernel_size=1, stride=1, use_act=True, use_norm=True)
        self.model_conf_dict['exp_before_cls'] = {'in': input_channels, 'out': last_channel}

        pool_type = getattr(opts, "model.layer.global_pool", "mean")
        self.last_channel = last_channel
        self.pooling = GlobalPool(pool_type=pool_type, keep_dim=False)
        self.classifier = nn.Sequential()
        self.classifier.add_module(name="global_pool", module=GlobalPool(pool_type=pool_type, keep_dim=False))
        self.classifier.add_module(name="classifier_dropout", module=nn.Dropout(p=0.2, inplace=True))
        self.classifier.add_module(name="classifier_fc",
                                   module=LinearLayer(in_features=last_channel, out_features=num_classes, bias=True))

        self.model_conf_dict['cls'] = {'in': last_channel, 'out': num_classes}

    def _make_layer(self, opts, mv2_config, width_mult, input_channel, round_nearest=8):
        if not isinstance(mv2_config[0], list):
            mv2_config = [mv2_config]

        mv2_block = nn.Sequential()
        count = 0
        for t, c, n, s in mv2_config:
            output_channel = make_divisible(c * width_mult, round_nearest)

            for i in range(n):
                stride = s if i == 0 else 1

                layer = InvertedResidual(opts=opts, in_channels=input_channel, out_channels=output_channel,
                                         stride=stride, expand_ratio=t)
                mv2_block.add_module(name="mv2_s_{}_idx_{}".format(stride, count), module=layer)
                count += 1
                input_channel = output_channel
        return mv2_block, input_channel

    def extract_features(self, x: Tensor) -> Tensor:
        x = self.conv_1(x)
        x = self.layer_1(x)

        # these are the two layers where we connect the input
        x = self.layer_2(x)
        x = self.layer_3(x)

        x = self.layer_4(x)
        x = self.layer_5(x)
        x = self.conv_1x1_exp(x)
        x = self.pooling(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        x = self.extract_features(x)
        try:
            x = self.classifier(x)
            return x
        except AttributeError:
            return x
        else:
            return x


if __name__ == '__main__':
    import torch

    inp = torch.Tensor(1, 3, 224, 224)
    model = MobileNetV2(opts=None)
    del(model.classifier)
    out = model(inp)
    print(out.shape)
    # wts_loc = "mv2_s1.0_imagenet.pt"
    # state_dict = torch.load(wts_loc, map_location='cpu')
    # model.load_state_dict(state_dict)
    # out = model(inpw
    # print(out.shape)
