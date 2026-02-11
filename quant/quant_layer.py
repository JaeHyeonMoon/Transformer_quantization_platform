import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union
from quant.quantizer import UniformAffineQuantizer, Log2Quantizer, UpperBitBoundQuantizer, UpperBitBoundQuantizer_attn
import os

class QuantModule(nn.Module):
    """
    Quantized Module that can perform quantized convolution or normal convolution.
    To activate quantization, please use set_quant_state function.
    """
    def __init__(self, org_module: Union[nn.Conv2d, nn.Linear], weight_quant_params: dict = {},
                 act_quant_params: dict = {}, n_groups: int = 1, dynamic_quant: bool = True, post_gelu: bool = False):
        super(QuantModule, self).__init__()
        if isinstance(org_module, nn.Conv2d): # Start of the layer
            self.fwd_kwargs = dict(stride=org_module.stride, padding=org_module.padding,
                                   dilation=org_module.dilation, groups=org_module.groups)
            self.fwd_func = F.conv2d
            self.disable_act_quant = True
        else:
            self.fwd_kwargs = dict()
            self.fwd_func = F.linear
            self.disable_act_quant = False

        self.weight = org_module.weight
        self.org_weight = org_module.weight.data.clone()
        self.act_quant_params = act_quant_params
        self.dynamic_quant = dynamic_quant
        if org_module.bias is not None:
            self.bias = org_module.bias
            self.org_bias = org_module.bias.data.clone()
        else:
            self.bias = None
            self.org_bias = None

        # de-activate the quantized forward default
        self.use_weight_quant = False
        self.use_act_quant = False
        self.n_groups = n_groups
        self.act_quant_params = act_quant_params
        
        self.weight_quantizer = UniformAffineQuantizer(**weight_quant_params)

        self.inited = False
        self.static = True

        if not self.disable_act_quant:
            if self.dynamic_quant:
                self.act_quantizer = UpperBitBoundQuantizer(n_groups = n_groups, post_gelu = post_gelu, **act_quant_params)
            else:
                self.act_quantizer = UniformAffineQuantizer(n_groups = n_groups, **act_quant_params)

        self.ignore_reconstruction = False

    def forward(self, input: torch.Tensor):
        if self.use_weight_quant:
            weight = self.weight_quantizer(self.weight)
            bias = self.bias
        else:
            weight = self.org_weight
            bias = self.org_bias

        input_f = input.clone()

        if self.use_act_quant and not self.disable_act_quant:
            input = self.act_quantizer(input)
        
        out = self.fwd_func(input, weight, bias, **self.fwd_kwargs)

        return out

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant

# For matmul operations
class QuantMatMul(nn.Module):
    def __init__(self, mode: str = 'matmul1', matmul_params: dict = {}, n_groups: int = 1):
        super(QuantMatMul, self).__init__()
        if mode == 'matmul1':
            A_quantizer = UpperBitBoundQuantizer_attn
        elif mode == 'matmul2':
            A_quantizer = Log2Quantizer
        else:
            raise ValueError('Unsupported mode')
            
        self.mode = mode
        self.A_quantizer = A_quantizer(n_groups = n_groups, **matmul_params)
        self.B_quantizer = UpperBitBoundQuantizer_attn(**matmul_params)

        self.ignore_reconstruction = False
        self.disable_act_quant = False

    def forward(self, A: torch.Tensor, B: torch.Tensor):
        if self.use_A_quant:
            A = self.A_quantizer(A)
        if self.use_B_quant:
            B = B.transpose(2,3)
            B = self.B_quantizer(B)
            B = B.transpose(2,3)
        out = A @ B

        return out

    def set_quant_state(self, A_quant: bool = False, B_quant: bool = False):
        self.use_A_quant = A_quant
        self.use_B_quant = B_quant