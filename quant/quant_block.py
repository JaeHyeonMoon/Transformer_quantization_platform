import torch
import torch.nn as nn
import torch.nn.functional as F

from quant.quant_layer import QuantModule, QuantMatMul
from timm.models.vision_transformer import Attention, Block
from timm.models.swin_transformer import window_partition, window_reverse, WindowAttention, PatchMerging, SwinTransformerBlock
from timm.models.layers.mlp import Mlp
import os

class BaseQuantBlock(nn.Module):
    """
    Base implementation of block structures for all networks.
    Due to the branch architecture, we have to perform activation function
    and quantization after the elemental-wise add operation, therefore, we
    put this part in this class.
    """
    def __init__(self):
        super().__init__()
        self.use_weight_quant = False
        self.use_act_quant = False

        self.ignore_reconstruction = False

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant 
        self.use_act_quant = act_quant
        for m in self.modules():
            if isinstance(m, QuantModule):
                m.set_quant_state(weight_quant, act_quant)
            elif isinstance(m, QuantMatMul):
                m.set_quant_state(act_quant, act_quant)

### Sub-blocks ###
class QAttention(BaseQuantBlock):
    def __init__(self, attention_layer: Attention, weight_quant_params: dict = {}, act_quant_params: dict = {}, matmul_params: dict = {}, n_groups: int = 1):
        super().__init__()
        self.num_heads = attention_layer.num_heads
        self.scale = attention_layer.scale
        self.n_bits = act_quant_params['n_bits']
        self.qkv = QuantModule(attention_layer.qkv, weight_quant_params, act_quant_params, n_groups, True)

        self.attn_drop = attention_layer.attn_drop

        self.matmul1 = QuantMatMul('matmul1', matmul_params, n_groups)
        self.matmul2 = QuantMatMul('matmul2', matmul_params, n_groups)

        act_quant_params1 = act_quant_params.copy()
        act_quant_params1['dim'] = None

        self.proj = QuantModule(attention_layer.proj, weight_quant_params, act_quant_params, n_groups, True)
        
        self.proj_drop = attention_layer.proj_drop

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) # [b,s,3,h,c]->[3,b,h,s,c]
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
        attn = self.matmul1(q, k.transpose(-2, -1)) * self.scale # [b,h,s,c],[b,h,c,s]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = self.matmul2(attn, v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class QWindowAttention(BaseQuantBlock):
    def __init__(self, attention_layer: WindowAttention, weight_quant_params: dict = {}, act_quant_params: dict = {}, matmul_params: dict = {}, n_groups: int = 1):
        super().__init__()
        self.dim = attention_layer.dim
        self.window_size = attention_layer.window_size
        self.window_area = attention_layer.window_area
        self.num_heads = attention_layer.num_heads
        self.scale = attention_layer.scale
        self.relative_position_bias_table = attention_layer.relative_position_bias_table
        self.relative_position_index = attention_layer.relative_position_index

        self.qkv = QuantModule(attention_layer.qkv, weight_quant_params, act_quant_params, n_groups, True)
        self.attn_drop = attention_layer.attn_drop

        self.matmul1 = QuantMatMul('matmul1', matmul_params, n_groups)
        self.matmul2 = QuantMatMul('matmul2', matmul_params, n_groups)

        self.proj = QuantModule(attention_layer.proj, weight_quant_params, act_quant_params, n_groups, True)
        self.proj_drop = attention_layer.proj_drop

        self.softmax = attention_layer.softmax

    def _get_rel_pos_bias(self):
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        return relative_position_bias.unsqueeze(0)

    def forward(self, x, mask = None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = self.matmul1(q, k.transpose(-2,-1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = self.matmul2(attn, v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class QPatchMerging(BaseQuantBlock):
    def __init__(self, merging_layer: PatchMerging, weight_quant_params: dict = {}, act_quant_params: dict = {}, matmul_params: dict = {}, n_groups: int = 1):
        super().__init__()
        self.input_resolution = merging_layer.input_resolution
        self.norm = merging_layer.norm
        self.reduction = QuantModule(merging_layer.reduction, weight_quant_params, act_quant_params, n_groups, True)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

class QMlp(BaseQuantBlock):
    def __init__(self, mlp_layer: Mlp, weight_quant_params: dict = {}, act_quant_params: dict = {}, matmul_params: dict = {}, n_groups: int = 1):
        super().__init__()
        self.fc1 = QuantModule(mlp_layer.fc1, weight_quant_params, act_quant_params, n_groups, True, False)

        self.act = mlp_layer.act
        self.drop1 = mlp_layer.drop1

        self.fc2 = QuantModule(mlp_layer.fc2, weight_quant_params, act_quant_params, n_groups, True, True)
        self.drop2 = mlp_layer.drop2

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

### Blocks ###
class QuantBlock(BaseQuantBlock):
    def __init__(self, block: Block, weight_quant_params: dict = {}, act_quant_params: dict = {}, matmul_params: dict = {}, n_groups: int = 1):
        super().__init__()
        self.norm1 = block.norm1
        self.attn = QAttention(block.attn, weight_quant_params, act_quant_params, matmul_params, n_groups)
        self.ls1 = block.ls1
        self.drop_path1 = block.drop_path1

        self.norm2 = block.norm2
        self.mlp = QMlp(block.mlp, weight_quant_params, act_quant_params, matmul_params, n_groups)
        self.ls2 = block.ls2
        self.drop_path2 = block.drop_path2

    def forward(self, x):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x

class QuantSwinTransformerBlock(BaseQuantBlock):
    def __init__(self, block: SwinTransformerBlock, weight_quant_params: dict = {}, act_quant_params: dict = {}, matmul_params: dict = {}, n_groups: int = 1):
        super().__init__()
        self.dim = block.dim
        self.input_resolution = block.input_resolution
        self.window_size = block.window_size
        self.shift_size = block.shift_size
        self.mlp_ratio = block.mlp_ratio

        self.norm1 = block.norm1
        self.attn = QWindowAttention(block.attn, weight_quant_params, act_quant_params, matmul_params, n_groups)

        self.drop_path = block.drop_path
        self.norm2 = block.norm2
        self.mlp = QMlp(block.mlp, weight_quant_params, act_quant_params, matmul_params, n_groups)
        self.attn_mask = block.attn_mask

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # num_win*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # num_win*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # num_win*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

subblock_specials = {
    Attention: QAttention,
    WindowAttention: QWindowAttention,
    PatchMerging: QPatchMerging,
    Mlp: QMlp
}

block_specials = {
    Block: QuantBlock,
    SwinTransformerBlock: QuantSwinTransformerBlock,
    PatchMerging: QPatchMerging
}