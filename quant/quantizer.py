import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from scipy.stats import norm

coeff = 66
percent = 66
constraint = 0.8
epsilon = 1e-4
on = True # on => False, no MPQ
swin_on = False
coco = 0.1

class StraightThrough(nn.Module):
    def __init__(self, channel_num: int = 1):
        super().__init__()

    def forward(self, input):
        return input

def round_ste(x: torch.Tensor):
    """
    Implement Straight-Through Estimator for rounding operation. 
    """
    return (x.round() - x).detach() + x

def lp_loss(pred, tgt, p=2.0, reduction='none'):
    """
    loss function measured in L_p Norm
    """
    if reduction == 'none':
        return (pred-tgt).abs().pow(p).sum(1).mean()
    else:
        return (pred-tgt).abs().pow(p).mean()

class UniformAffineQuantizer(nn.Module):
    def __init__(self, n_bits: int = 8, dim: int = None, scale_method: tuple = ('max', 0), leaf_param: bool = False, n_groups: int = 1):
        super(UniformAffineQuantizer, self).__init__()
        assert 2 <= n_bits <= 8, 'bitwidth not supported'
        self.n_bits = n_bits
        self.n_levels = 2 ** self.n_bits
        self.dim = dim
        self.scale_method, self.eps = scale_method
        self.leaf_param = leaf_param
        self.n_groups = n_groups
        if self.n_groups > 1:
            self.group_indices = None
        self.delta = None
        self.zero_point = None
        self.inited = False

    def forward(self, x: torch.Tensor):
        if self.n_bits >= 8:
            return x

        if self.inited is False:
            if self.leaf_param:
                delta, self.zero_point = self.init_quantization_scale(x, self.dim)
                self.delta = torch.nn.Parameter(delta)
            else:
                self.delta, self.zero_point = self.init_quantization_scale(x, self.dim)
            self.inited = True

        # start quantization
        x_int = round_ste(x / self.delta) + self.zero_point
        x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
        x_dequant = (x_quant - self.zero_point) * self.delta
        return x_dequant

    def init_quantization_scale(self, x: torch.Tensor, dim: int = None):
        x_clone = x.clone().detach()
        device = x.device
        
        if dim is not None:
            n_quant = x_clone.shape[dim]
            delta = torch.zeros(n_quant).type_as(x)
            zero_point = torch.zeros(n_quant).type_as(x)
            for c in range(n_quant):
                delta[c], zero_point[c] = self.init_quantization_scale(torch.index_select(x_clone, dim, torch.tensor([c]).to(device)), dim=None)
            dims = [1 for _ in range(x_clone.ndim)]
            dims[dim] = -1
            delta = delta.view(*dims)
            zero_point = zero_point.view(*dims)

        else:
            if self.scale_method == 'max':
                x_min = min(x.min().item(), 0)
                x_max = max(x.max().item(), 0)
            elif self.scale_method == 'percentile':
                if x.numel() > 1000000:
                    sample_ratio = 1000000/x.numel()
                    x_sample = torch.randn_like(x) > float(norm.ppf(1 - sample_ratio))
                    x_sample = x[x_sample]
                else:
                    x_sample = x

                eps = torch.tensor([1-self.eps, self.eps]).to(device)
                x_max, x_min = torch.quantile(x_sample, eps)
            elif self.scale_method == 'std':
                mean, std = x.mean().item(), x.std().item()
                x_max = min(mean+self.eps*std, x.max().item())
                x_min = max(mean-self.eps*std, x.min().item())
            else:
                raise NotImplementedError

            delta = float(x_max - x_min) / (self.n_levels - 1)
            if delta < 1e-10:
                warnings.warn('Quantization range close to zero: [{}, {}]'.format(x_min, x_max))
                delta = 1e-10
                
            delta = torch.tensor(delta).type_as(x)
            zero_point = (-x_min / delta).round()

        return delta, zero_point

    def bitwidth_refactor(self, refactored_bit: int):
        assert 2 <= refactored_bit <= 8, 'bitwidth not supported'
        self.n_bits = refactored_bit
        self.n_levels = 2 ** self.n_bits


class UpperBitBoundQuantizer_attn(nn.Module): # Dynamic token-wise quantization
    def __init__(self, n_bits: int = 8, dim: int = None, scale_method: tuple = ('max', 0), leaf_param: bool = False, n_groups: int = 1):
        super(UpperBitBoundQuantizer_attn, self).__init__()
        assert 2 <= n_bits <= 8, 'bitwidth not supported'
        self.n_bits = n_bits
        self.n_levels = 2 ** self.n_bits
        self.n_bits_fix = self.n_bits * 1
        self.dim = -1
        self.batch_size = 32
        self.size_div_batch_size = None
        self.token_num = 0
        self.channel_num = 0
        self.n_groups = n_groups
        self.inited = True
        self.scale = None
        self.alpha = None
        self.coeff = None
        self.mean = None
        self.constraint = None
        self.error = None
        self.std = None
        self.threshold = None
        self.continue_ = True
        self.window = True
        
    def forward(self, x: torch.Tensor):
        if x.ndim == 2: # classifier
            return x
        b, h, t, c = x.size(0), x.size(1), x.size(2), x.size(3)
        
        x_ori = x.clone()
        self.batch_size = b
        if self.size_div_batch_size is None:
            self.size_div_batch_size = x.size(0) // self.batch_size
        global swin_on
        if swin_on:
            batch_num = b / 32
            if batch_num == 1:
                self.window = False
            if batch_num > 1 and batch_num % 1 == 0:
                x = x.view(32, int(batch_num), h, t, c) # 32, batch_num, h, t, c
                self.num_sqrt = int(math.sqrt(batch_num))
                self.num_div = int(self.num_sqrt / 2)
                x = x.view(32, self.num_sqrt, self.num_sqrt, h, t, c)
                x = x.view(32, 2, self.num_div, 2, self.num_div, h, t, c).permute(0,2,4,1,3,5,6,7)
                x = x.reshape(32, self.num_div ** 2, 2**2, h, t, c).transpose(2,3)
                x = x.reshape(32, self.num_div ** 2 * h, 2**2 * t, c)

                t_ori = t
                h_ori = h
                t = 2**2 * t
                h = self.num_div ** 2 * h

        if self.inited:
            self.coeff = (x.std() / (x.max() - x.min()))
            self.std = x.std()
            self.mean = x.median()
            x_max = torch.quantile(x, 1-1e-1, dim=1, keepdim=True)
            x_max = torch.quantile(x_max, 1-1e-1, dim=2, keepdim=True)
            x_max = torch.quantile(x_max, 1-1e-1, dim=3, keepdim=True)
            
            for num in range(0, 21):
                for num_ in range(0, 21):
                    constraint = (num) / 5
                    threshold = num_ / 5 + ((2**(self.n_bits_fix + 1) - 1)/(2**self.n_bits_fix - 2))

                    x_dequant_init = self.init_quant_out(x, constraint, threshold)
                    error = (x - x_dequant_init).abs().mean()
                    if self.error == None:
                        self.error = error
                        self.constraint = constraint
                        self.threshold = threshold
                    elif self.error >= error:
                        self.error = error
                        self.constraint = constraint
                        self.threshold = threshold
            self.inited = False

        b_, h_, t_, c_ = x.size()
        x = x.transpose(1,2).reshape(b_,t_,h_*c_)
        self.token_num = x.size(1)
        self.channel_num = x.size(2)
        error_ulg = (x.max(dim=2)[0] - x.min(dim=2)[0])

        attn_sum = 1 / error_ulg
        attn_sum_ = error_ulg
        attn_std = (x < (self.mean + error_ulg.unsqueeze(2) * self.coeff * self.constraint)).sum(dim=2).float() - (x < (self.mean - error_ulg.unsqueeze(2) * self.coeff * self.constraint)).sum(dim=2).float()

        global coco
        if self.constraint == 0:
            attn_std = torch.clamp(attn_std, min=1e-6)
        else:
            attn_std = torch.clamp(attn_std, min=self.channel_num * coco)
        
        attn_std = 1 / attn_std
        attn_std[attn_std>1] = 0

        attn_sort_dec, dec_idx = attn_sum_.sort(dim=1, descending=True)
        attn_sort_inc, inc_idx = attn_sum_.sort(dim=1)
        attn_sort_ = ((attn_sort_dec * attn_std.gather(1, dec_idx)) - (attn_sort_inc * attn_std.gather(1, inc_idx) * self.threshold)) / self.std
        attn_sort_ = (attn_sort_ > 0)

        min_indices = torch.full((attn_sort_.shape[0],), -1, dtype=torch.long).cuda()

        for i in range(attn_sort_.shape[0]):
            row = attn_sort_[i]
            zeros = (row == 0).nonzero()
            if len(zeros) > 0:
                min_indices[i] = zeros[0,0]
        
        attn_sort_diff = min_indices.unsqueeze(1)
        attn_sort = attn_sum.sort(dim=1)[1].sort(dim=1)[1]

        bit_token = torch.zeros_like(attn_sort)
        global on
        if on:
            bit_token[attn_sort < attn_sort_diff] = self.n_bits + 1
            bit_token[(attn_sort >= attn_sort_diff) & (attn_sort <= ((t-1)-attn_sort_diff))] = self.n_bits
            bit_token[attn_sort > ((t-1)-attn_sort_diff)] = self.n_bits - 1
        else:
            bit_token[attn_sort < attn_sort_diff] = self.n_bits
            bit_token[(attn_sort >= attn_sort_diff) & (attn_sort <= ((t-1)-attn_sort_diff))] = self.n_bits
            bit_token[attn_sort > ((t-1)-attn_sort_diff)] = self.n_bits

        self.n_levels = 2 ** bit_token
        self.n_levels = self.n_levels.unsqueeze(1).unsqueeze(3)
        x = x.view(-1, self.size_div_batch_size, *x.size()[1:])
        delta, zero_point = self.init_quantization_scale(x, self.dim)

        # start quantization
        x_int = round_ste(x / delta) + zero_point
        x_int = x_int / (self.n_levels - 1)
        x_quant = torch.clamp(x_int, 0, 1)
        x_quant = x_quant * (self.n_levels - 1)
        x_dequant = (x_quant - zero_point) * delta

        if swin_on:
            if batch_num > 1 and batch_num % 1 == 0:
                x_dequant = x_dequant.squeeze().reshape(32, t, h, c).transpose(1,2) # 32, h, t, c
                x_dequant = x_dequant.reshape(32, self.num_div ** 2, h_ori, 2**2, t_ori, c).transpose(2,3)
                x_dequant = x_dequant.reshape(32, self.num_div, self.num_div, 2, 2, h_ori, t_ori, c).permute(0,3,1,4,2,5,6,7)
                x_dequant = x_dequant.reshape(32, self.num_sqrt, self.num_sqrt, h_ori, t_ori, c)
                x_dequant = x_dequant.reshape(32, int(batch_num), h_ori, t_ori, c)
                
            else:
                x_dequant = x_dequant.squeeze().reshape(b, t, h, c).transpose(1,2)
        else:    
            x_dequant = x_dequant.squeeze().reshape(b, t, h, c).transpose(1,2)
        if swin_on:
            if batch_num > 1 and batch_num % 1 == 0:
                x_dequant = x_dequant.reshape(32 * int(batch_num), h_ori, t_ori, c)

        return x_dequant

    def init_quant_out(self, x, constraint, threshold):
        b, h, t, c = x.size(0), x.size(1), x.size(2), x.size(3)
        x = x.transpose(1,2).reshape(b,t,h*c)
        self.token_num = x.size(1)
        self.channel_num = x.size(2)
        error_ulg = (x.max(dim=2)[0] - x.min(dim=2)[0])
        attn_sum = 1 / error_ulg
        attn_sum_ = error_ulg
        attn_std = (x < (self.mean + error_ulg.unsqueeze(2) * self.coeff * constraint)).sum(dim=2).float() - (x < (self.mean - error_ulg.unsqueeze(2) * self.coeff * constraint)).sum(dim=2).float()
        
        global coco
        if constraint == 0:
            attn_std = torch.clamp(attn_std, min=1e-6)
        else:
            attn_std = torch.clamp(attn_std, min=coco * self.channel_num)

        attn_std = 1 / attn_std
        attn_std[attn_std>1] = 0

        attn_sort_dec, dec_idx = attn_sum_.sort(dim=1, descending=True)
        attn_sort_inc, inc_idx = attn_sum_.sort(dim=1)
        attn_sort_ = ((attn_sort_dec * attn_std.gather(1, dec_idx)) - (attn_sort_inc * attn_std.gather(1, inc_idx) * threshold)) / self.std
        attn_sort_ = (attn_sort_ > 0)

        min_indices = torch.full((attn_sort_.shape[0],), -1, dtype=torch.long).cuda()

        for i in range(attn_sort_.shape[0]):
            row = attn_sort_[i]
            zeros = (row == 0).nonzero()
            if len(zeros) > 0:
                min_indices[i] = zeros[0,0]
        
        attn_sort_diff = min_indices.unsqueeze(1)
        attn_sort = attn_sum.sort(dim=1)[1].sort(dim=1)[1]

        bit_token = torch.zeros_like(attn_sort)

        bit_token[attn_sort < attn_sort_diff] = self.n_bits + 1
        bit_token[(attn_sort >= attn_sort_diff) & (attn_sort <= ((t-1)-attn_sort_diff))] = self.n_bits
        bit_token[attn_sort > ((t-1)-attn_sort_diff)] = self.n_bits - 1

        self.n_levels = 2 ** bit_token
        self.n_levels = self.n_levels.unsqueeze(1).unsqueeze(3)
        x = x.view(-1, self.size_div_batch_size, *x.size()[1:])
        delta, zero_point = self.init_quantization_scale(x, self.dim)
        
        x_int = round_ste(x / delta) + zero_point
        x_int = x_int / (self.n_levels - 1)
        x_quant = torch.clamp(x_int, 0, 1)
        x_quant = x_quant * (self.n_levels - 1)
        x_dequant = (x_quant - zero_point) * delta
        x_dequant = x_dequant.squeeze().reshape(b, t, h, c).transpose(1,2)
        return x_dequant

    def init_quantization_scale(self, x: torch.Tensor, dim: int = None):
        x_clone = x.clone().detach()
        x_min = x_clone.amin(dim=(1,3), keepdim=True)
        x_max = x_clone.amax(dim=(1,3), keepdim=True)
        delta = (x_max - x_min) / (self.n_levels - 1)
        delta[delta < 1e-8] = 1e-8
        zero_point = (-x_min / delta).round()
        return delta, zero_point

    def bitwidth_refactor(self, refactored_bit: int):
        assert 2 <= refactored_bit <= 8, 'bitwidth not supported'
        self.n_bits = refactored_bit
        self.n_levels = 2 ** self.n_bits


class UpperBitBoundQuantizer(nn.Module): # Dynamic token-wise quantization
    def __init__(self, n_bits: int = 8, dim: int = None, scale_method: tuple = ('max', 0), leaf_param: bool = False, n_groups: int = 1, post_gelu: bool = False):
        super(UpperBitBoundQuantizer, self).__init__()
        assert 2 <= n_bits <= 8, 'bitwidth not supported'
        self.n_bits = n_bits
        self.n_levels = 2 ** self.n_bits
        self.n_bits_fix = self.n_bits * 1
        self.dim = -1
        self.batch_size = 32
        self.size_div_batch_size = None
        self.token_num = 0
        self.channel_num = 0
        self.n_groups = n_groups
        self.register_buffer('init', torch.tensor(1))
        self.inited= True
        self.scale = None
        self.alpha = None
        self.coeff = None
        self.mean = None
        self.constraint = None
        self.error = None
        self.x_dequant = None
        self.std = None
        self.continue_ = True
        self.window = True
        self.mean_thres = 0
        self.threshold = 0
        self.threshold_ = 0
        self.post_gelu = post_gelu

    def forward(self, x: torch.Tensor):
        if x.ndim == 2: # classifier
            return x
        self.batch_size, self.token_num, self.channel_num = x.size()[:3]
        
        if self.size_div_batch_size is None:
            self.size_div_batch_size = x.size(0) // self.batch_size

        global epsilon
        if self.inited: # delta, beta, c에 대한 calibration
            self.coeff = (x.std() / (x.max() - x.min()))
            self.std = x.std()
            self.mean = x.median()
            x_max = torch.quantile(x, 1-1e-1, dim=1, keepdim=True)
            x_max = torch.quantile(x_max, 1-1e-1, dim=2, keepdim=True)

            if self.post_gelu: # Post-GELU
                for num in range(0, 21):
                    for num_ in range(0, 21):
                        for num__ in range(0, 11):
                            constraint = num / 5 # beta
                            threshold = num_ / 5 + ((2**(self.n_bits_fix + 1) - 1)/(2**self.n_bits_fix - 2)) # c
                            threshold_ = num__ / 10 # shifting quantizer hyper-parameter

                            x_dequant_init = self.init_quant_out(x, constraint, threshold, threshold_)
                            error = (x - x_dequant_init).abs().mean()
                            if self.error == None:
                                self.error = error
                                self.constraint = constraint
                                self.threshold = threshold
                                self.threshold_ = threshold_
                            elif self.error >= error:
                                self.error = error
                                self.constraint = constraint # beta
                                self.threshold = threshold # c
                                self.threshold_ = threshold_ # delta
            else:
                for num in range(0, 21):
                    for num_ in range(0, 21):
                        constraint = (num) / 5
                        threshold = num_ / 5 + ((2**(self.n_bits_fix + 1) - 1)/(2**self.n_bits_fix - 2))

                        x_dequant_init = self.init_quant_out(x, constraint, threshold)
                        error = (x - x_dequant_init).abs().mean()
                        
                        if self.error == None:
                            self.error = error
                            self.constraint = constraint
                            self.threshold = threshold
                        elif self.error >= error:
                            self.error = error
                            self.constraint = constraint # beta
                            self.threshold = threshold # c
            self.inited = False

        error_ulg = (x.max(dim=2)[0] - x.min(dim=2)[0])
        error_ulg[:,0] = 1e+8 * error_ulg[:,0]

        attn_std = (x < (self.mean + error_ulg.unsqueeze(2) * self.coeff * self.constraint)).sum(dim=2).float() - (x < (self.mean - error_ulg.unsqueeze(2) * self.coeff * self.constraint)).sum(dim=2).float()
        global coco
        
        if self.constraint > 0:
            attn_std = torch.clamp(attn_std, min=coco * self.channel_num)
        else:
            attn_std = torch.clamp(attn_std, min=1e-6)
        attn_std = (1 / attn_std)
        attn_std[attn_std>1] = 0

        attn_sort_inc, inc_idx = error_ulg.sort(dim=1)
        attn_sort_dec, dec_idx = torch.flip(attn_sort_inc, dims=[1]), torch.flip(inc_idx, dims=[1])
        attn_sort_ = ((attn_sort_dec * attn_std.gather(1, dec_idx)) - (attn_sort_inc * attn_std.gather(1, inc_idx) * self.threshold)) / self.std
        min_indices = attn_sort_[:, :self.token_num//2].cumsum(dim=1).argmax(dim=1)
        
        inst, token = (attn_sort_ < 0).nonzero(as_tuple=True)
        inst = inst.unique_consecutive(return_counts=True)[1].cumsum(dim=0)[:-1]
        min_indices = torch.cat([token[0:1], token[inst]], dim=0)
        if attn_std.abs().sum() == 0:
            min_indices = torch.zeros(attn_std.size(0)).to(attn_std.device)
        
        attn_sort_diff = min_indices.unsqueeze(1)
        attn_sort_diff = torch.clamp(attn_sort_diff-1, min=0).cuda()
        attn_sort = error_ulg.argsort(dim=1, descending=True).argsort(dim=1)

        bit_token = torch.zeros(self.batch_size, self.token_num).to(x.device)

        global on
        if on: # to use MPQ
            bit_token[attn_sort <= attn_sort_diff] = self.n_bits + 1
            bit_token[(attn_sort > attn_sort_diff) & (attn_sort <= ((self.token_num-2)-attn_sort_diff))] = self.n_bits
            bit_token[attn_sort > ((self.token_num-2)-attn_sort_diff)] = self.n_bits - 1
        else:
            bit_token.fill_(self.n_bits)
        
        self.n_levels = 2 ** bit_token
        self.n_levels = self.n_levels.unsqueeze(1).unsqueeze(3)
            
        x = x.view(-1, self.size_div_batch_size, *x.size()[1:])

        if self.post_gelu:
            delta, zero_point, bit_shift = self.init_quantization_scale_shift(x, self.dim, self.threshold_)

            x = x.view(*x.size()[:-1], self.n_groups, -1)
            x = x * (2**(bit_shift * (x<0)))
            x_int = round_ste(x / delta) + zero_point
            x_int = x_int / (self.n_levels.unsqueeze(3) - 1)
            x_quant = torch.clamp(x_int, 0, 1)
            x_quant = x_quant * (self.n_levels.unsqueeze(3) - 1)
            x_dequant = (x_quant - zero_point) * delta                
            x_dequant = x_dequant * (2**(-bit_shift * (x<0)))
            x_dequant = x_dequant.view(-1, *x_dequant.size()[2:])
        else:
            delta, zero_point = self.init_quantization_scale(x, self.dim)

            x = x.view(*x.size()[:-1], self.n_groups, -1)
            x_int = round_ste(x / delta) + zero_point
            x_int = x_int / (self.n_levels.unsqueeze(3) - 1)
            x_quant = torch.clamp(x_int, 0, 1)
            x_quant = x_quant * (self.n_levels.unsqueeze(3) - 1)
            x_dequant = (x_quant - zero_point) * delta
            x_dequant = x_dequant.view(-1, *x_dequant.size()[2:])

        return x_dequant.flatten(-2, -1)

    def init_quant_out(self, x, constraint, threshold, threshold_=1.0):
        error_ulg = (x.max(dim=2)[0] - x.min(dim=2)[0])
        error_ulg[:,0] = 1e+8 * error_ulg[:,0]

        attn_sum = 1 / error_ulg
        attn_sum_ = error_ulg
        attn_std = (x < (self.mean + error_ulg.unsqueeze(2) * self.coeff * constraint)).sum(dim=2).float() - (x < (self.mean - error_ulg.unsqueeze(2) * self.coeff * constraint)).sum(dim=2).float()

        channel_num_ = x.size(2)
        global coco

        if constraint > 0:
            attn_std = torch.clamp(attn_std, min=coco * channel_num_)
        else:
            attn_std = torch.clamp(attn_std, min=1e-6)

        attn_std = 1 / attn_std
        attn_std[attn_std>1] = 0
        attn_sort_dec, dec_idx = attn_sum_.sort(dim=1, descending=True)
        attn_sort_inc, inc_idx = attn_sum_.sort(dim=1)
        attn_sort_ = ((attn_sort_dec * attn_std.gather(1, dec_idx)) - (attn_sort_inc * attn_std.gather(1, inc_idx) * threshold))
        attn_sort_ = (attn_sort_ > 0)

        min_indices = torch.full((attn_sort_.shape[0],), -1, dtype=torch.long).cuda()

        for i in range(attn_sort_.shape[0]):
            row = attn_sort_[i]
            zeros = (row == 0).nonzero()
            if len(zeros) > 0:
                min_indices[i] = zeros[0,0]

        attn_sort_diff = min_indices.unsqueeze(1)
        attn_sort_diff = torch.clamp(attn_sort_diff-1, min=0).cuda()

        attn_sort = attn_sum.sort(dim=1)[1].sort(dim=1)[1]
        bit_token = torch.zeros_like(attn_sort)

        global on
        if on:
            bit_token[attn_sort <= attn_sort_diff] = self.n_bits + 1
            bit_token[(attn_sort > attn_sort_diff) & (attn_sort <= ((self.token_num-2)-attn_sort_diff))] = self.n_bits
            bit_token[attn_sort > ((self.token_num-2)-attn_sort_diff)] = self.n_bits - 1
        else:
            bit_token[attn_sort < attn_sort_diff] = self.n_bits
            bit_token[(attn_sort >= attn_sort_diff) & (attn_sort <= ((self.token_num-1)-attn_sort_diff))] = self.n_bits
            bit_token[attn_sort > ((self.token_num-1)-attn_sort_diff)] = self.n_bits
    
        self.n_levels = 2 ** bit_token
        self.n_levels = self.n_levels.unsqueeze(1).unsqueeze(3)
            
        x = x.view(-1, self.size_div_batch_size, *x.size()[1:])

        if self.post_gelu:
            delta, zero_point, bit_shift = self.init_quantization_scale_shift(x, self.dim, threshold_)

            x = x.view(*x.size()[:-1], self.n_groups, -1)
            x = x * (2**(bit_shift * (x<0)))
            x_int = round_ste(x / delta) + zero_point
            x_int = x_int / (self.n_levels.unsqueeze(3) - 1)
            x_quant = torch.clamp(x_int, 0, 1)
            x_quant = x_quant * (self.n_levels.unsqueeze(3) - 1)
            x_dequant = (x_quant - zero_point) * delta
            x_dequant = x_dequant * (2**(-bit_shift * (x<0)))
        
        else:
            delta, zero_point = self.init_quantization_scale(x, self.dim)

            x = x.view(*x.size()[:-1], self.n_groups, -1)
            x_int = round_ste(x / delta) + zero_point
            x_int = x_int / (self.n_levels.unsqueeze(3) - 1)
            x_quant = torch.clamp(x_int, 0, 1)
            x_quant = x_quant * (self.n_levels.unsqueeze(3) - 1)
            x_dequant = (x_quant - zero_point) * delta

        x_dequant = x_dequant.view(-1, *x_dequant.size()[2:]).flatten(-2, -1)
        return x_dequant

    def init_quantization_scale(self, x: torch.Tensor, dim: int = None):
        x_clone = x.detach()

        B, one, S, C = x_clone.shape
        G = int(self.n_groups)
        assert C % G == 0, f"channel dim C={C} must be divisible by n_groups={G}"
        cg = C // G
        xg = x_clone.view(B, 1, S, G, cg)

        # 그룹 내부 채널(cg) 축에 대해 min/max (token-wise 유지)
        x_min = xg.amin(dim=-1, keepdim=True)  # [B,1,S,G,1]
        x_max = xg.amax(dim=-1, keepdim=True)  # [B,1,S,G,1]
        delta = (x_max - x_min) / (self.n_levels.unsqueeze(3) - 1)
        delta = torch.clamp(delta, min=1e-8)

        zero_point = (-x_min / delta).round()
        return delta, zero_point
        
    def init_quantization_scale_shift(self, x, dim, threshold):
        x_clone = x.detach()

        B, one, S, C = x_clone.shape
        G = int(self.n_groups)
        assert C % G == 0, f"channel dim C={C} must be divisible by n_groups={G}"

        cg = C // G
        xg = x_clone.view(B, 1, S, G, cg)

        x_min = xg.amin(dim=-1, keepdim=True)  # [B,1,S,G,1]
        x_max = xg.amax(dim=-1, keepdim=True)  # [B,1,S,G,1]

        eps = 1e-12
        denom = torch.where(x_min.abs() < eps, x_min.sign() * eps, x_min)
        ratio = torch.clamp((-threshold * x_max / denom), min=1.0)

        bit_shift = torch.clamp(torch.log2(ratio), min=0.0).round()  # [B,1,S,G,1]
        x_min_shifted = x_min * (2.0 ** bit_shift)

        delta = (x_max - x_min_shifted) / (self.n_levels.unsqueeze(3) - 1)
        delta = torch.clamp(delta, min=1e-8)

        zero_point = (-x_min_shifted / delta).round()
        return delta, zero_point, bit_shift

    def bitwidth_refactor(self, refactored_bit: int):
        assert 2 <= refactored_bit <= 8, 'bitwidth not supported'
        self.n_bits = refactored_bit
        self.n_levels = 2 ** self.n_bits


class Log2Quantizer(nn.Module):
    def __init__(self, n_bits: int = 8, dim: int = None, scale_method: tuple = ('max', 0), leaf_param: bool = False, n_groups: int = 1):
        super(Log2Quantizer, self).__init__()
        assert 2 <= n_bits <= 8, 'bitwidth not supported'
        self.n_bits = n_bits
        self.n_levels = 2 ** self.n_bits
        self.dim = dim
        self.scale_method, self.eps = scale_method
        self.leaf_param = leaf_param
        self.n_groups = n_groups
        if self.n_groups > 1:
            self.alpha = None
        self.delta = None
        self.inited = True
        self.batch_size = 32
        self.size_div_batch_size = None
        self.n_levels_fix = (2 ** n_bits) - 1
        self.error = None
        self.constraint = None
        self.threshold = None
        self.token_num = None
        self.window = True

    def forward(self, x):
        b, h, t, c = x.size(0), x.size(1), x.size(2), x.size(3)
        self.batch_size = b
        self.token_num = t

        if self.size_div_batch_size is None:
            self.size_div_batch_size = x.size(0) // self.batch_size

        global swin_on
        
        if swin_on:
            batch_num = b / 32
            if batch_num == 1:
                self.window = False
            if batch_num > 1 and batch_num % 1 == 0:
                x = x.view(32, int(batch_num), h, t, c) # 32, batch_num, h, t, c
                self.num_sqrt = int(math.sqrt(batch_num))
                self.num_div = int(self.num_sqrt / 2)
                x = x.view(32, self.num_sqrt, self.num_sqrt, h, t, c)
                x = x.view(32, 2, self.num_div, 2, self.num_div, h, t, c).permute(0,2,4,1,3,5,6,7)
                x = x.reshape(32, self.num_div ** 2, 2**2, h, t, c).transpose(2,3)
                x = x.reshape(32, self.num_div ** 2 * h, 2**2 * t, c)

                t_ori = t
                h_ori = h
                t = 2**2 * t
                h = self.num_div ** 2 * h

        if self.inited:
            self.coeff =  (x.std() / (x.max() - x.min()))
            self.mean = x.mean()
            x_max = torch.quantile(x, 1-1e-1, dim=1, keepdim=True)
            x_max = torch.quantile(x_max, 1-1e-1, dim=2, keepdim=True)
            x_max = torch.quantile(x_max, 1-1e-1, dim=3, keepdim=True)


            for num in range(0, 21):
                constraint = (num) / 20
                threshold = 1.0
                x_dequant_init = self.init_quant_out(x, constraint, threshold)
                error = (x - x_dequant_init).abs().mean()
                if self.error == None:
                    self.error = error
                    self.constraint = constraint
                    self.threshold = threshold
                elif self.error >= error:
                    self.error = error
                    self.constraint = constraint
                    self.threshold = threshold

            self.inited = False

        b_, h_, t_, c_ = x.size()
        x = x.transpose(1,2).reshape(b_, t_, h_*c_)

        error_ulg = (x.max(dim=2)[0])
        attn_sum = 1 / error_ulg
        attn_sum_ = error_ulg
        attn_std = (x < attn_sum_.unsqueeze(2) * (2**-(self.n_levels_fix))).sum(dim=2).float() - (x < attn_sum_.unsqueeze(2) * (2**-(self.n_levels_fix * (2**self.constraint)))).sum(dim=2).float()
        attn_std_down = (x < attn_sum_.unsqueeze(2) * (2**-(self.n_levels_fix/2))).sum(dim=2).float() - (x < attn_sum_.unsqueeze(2) * (2**-((self.n_levels_fix/2) * (2**self.constraint)))).sum(dim=2).float()

        attn_sort_dec, dec_idx = attn_sum_.sort(dim=1, descending=True)
        attn_sort_inc, inc_idx = attn_sum_.sort(dim=1)

        attn_sort_ = ((attn_sort_dec * attn_std.gather(1, dec_idx)) - (attn_sort_inc * attn_std_down.gather(1, inc_idx) * (2))) / x.std(dim=2).mean(dim=1, keepdim=True)
        attn_sort_diff = (attn_sort_[:,:int(t//2)-1]>0).float().sum(dim=1, keepdim=True)
        attn_sort = attn_sum.sort(dim=1)[1].sort(dim=1)[1]

        bit_token = torch.zeros_like(attn_sort)

        global on

        if on:
            bit_token[attn_sort < attn_sort_diff] = self.n_bits + 1
            bit_token[(attn_sort >= attn_sort_diff) & (attn_sort <= ((t-1) - attn_sort_diff))] = self.n_bits
            bit_token[attn_sort > ((t - 1) - attn_sort_diff)] = self.n_bits - 1

        else:
            bit_token[attn_sort <= attn_sort_diff] = self.n_bits
            bit_token[(attn_sort > attn_sort_diff) & (attn_sort <= ((self.token_num-2)-attn_sort_diff))] = self.n_bits
            bit_token[attn_sort > ((self.token_num-2)-attn_sort_diff)] = self.n_bits

        bit_token[attn_sort <= attn_sort_diff] = self.n_bits
        bit_token[(attn_sort > attn_sort_diff) & (attn_sort <= ((self.token_num-2)-attn_sort_diff))] = self.n_bits
        bit_token[attn_sort > ((self.token_num-2)-attn_sort_diff)] = self.n_bits

        self.n_levels = 2 ** bit_token

        self.n_levels = self.n_levels.unsqueeze(1).unsqueeze(3)

        x = x.view(-1, self.size_div_batch_size, *x.size()[1:])

        self.delta = self.init_quantization_scale(x, self.dim)
        x_int = round_ste(-torch.log2(torch.clamp(x / self.delta, min=1e-8)))

        mask = x_int >= self.n_levels

        x_int = x_int / (self.n_levels - 1)
        x_quant = torch.clamp(x_int, 0, 1)
        x_quant = x_quant * (self.n_levels - 1)
        x_dequant = self.delta * 2 ** (-x_quant)
        x_dequant[mask] = 0

        if swin_on:
            if batch_num > 1 and batch_num % 1 == 0:
                x_dequant = x_dequant.squeeze().reshape(32, t, h, c).transpose(1,2) # 32, h, t, c
                x_dequant = x_dequant.reshape(32, self.num_div ** 2, h_ori, 2**2, t_ori, c).transpose(2,3)
                x_dequant = x_dequant.reshape(32, self.num_div, self.num_div, 2, 2, h_ori, t_ori, c).permute(0,3,1,4,2,5,6,7)
                x_dequant = x_dequant.reshape(32, self.num_sqrt, self.num_sqrt, h_ori, t_ori, c)
                x_dequant = x_dequant.reshape(32, int(batch_num), h_ori, t_ori, c)
                
            else:
                x_dequant = x_dequant.squeeze().reshape(b, t, h, c).transpose(1,2)
        else:    
            x_dequant = x_dequant.squeeze().reshape(b, t, h, c).transpose(1,2)
        if swin_on:
            if batch_num > 1 and batch_num % 1 == 0:
                x_dequant = x_dequant.reshape(32 * int(batch_num), h_ori, t_ori, c)
        return x_dequant

    def init_quantization_scale(self, x: torch.Tensor, dim: int = None):
        x_clone = x.clone().detach()

        x_max = x_clone.amax(dim=(1,3), keepdim=True)
        return x_max

    def init_quant_out(self, x, constraint, threshold):
        b, h, t, c = x.size(0), x.size(1), x.size(2), x.size(3)
        x = x.transpose(1,2).reshape(b, t, h*c)

        error_ulg = (x.max(dim=2)[0])
        attn_sum = 1 / error_ulg
        attn_sum_ = error_ulg

        # Using the number
        attn_std = (x < attn_sum_.unsqueeze(2) * (2**-(self.n_levels_fix))).sum(dim=2).float() - (x < attn_sum_.unsqueeze(2) * (2**-(self.n_levels_fix*(2**constraint)))).sum(dim=2).float() 
        attn_std_down = (x < attn_sum_.unsqueeze(2) * (2**-(self.n_levels_fix/2))).sum(dim=2).float() - (x < attn_sum_.unsqueeze(2) * (2**-((self.n_levels_fix/2)*(2**constraint)))).sum(dim=2).float() 

        attn_sort_dec, dec_idx = attn_sum_.sort(dim=1, descending=True)
        attn_sort_inc, inc_idx = attn_sum_.sort(dim=1)
        attn_sort_ = ((attn_sort_dec * attn_std.gather(1, dec_idx)) - (attn_sort_inc * attn_std_down.gather(1, inc_idx) * (2))) / x.std(dim=2).mean(dim=1, keepdim=True)
        attn_sort_diff = (attn_sort_[:,:int(t//2)-1]>0).float().sum(dim=1, keepdim=True)
        attn_sort = attn_sum.sort(dim=1)[1].sort(dim=1)[1]

        bit_token = torch.zeros_like(attn_sort)

        global on
        if on:
            bit_token[attn_sort < attn_sort_diff] = self.n_bits + 1
            bit_token[(attn_sort >= attn_sort_diff) & (attn_sort <= ((t-1)-attn_sort_diff))] = self.n_bits
            bit_token[attn_sort > ((t-1)-attn_sort_diff)] = self.n_bits - 1
        else:
            bit_token[attn_sort <= attn_sort_diff] = self.n_bits
            bit_token[(attn_sort > attn_sort_diff) & (attn_sort <= ((self.token_num-1)-attn_sort_diff))] = self.n_bits
            bit_token[attn_sort > ((self.token_num-1)-attn_sort_diff)] = self.n_bits

        self.n_levels = 2 ** bit_token
        self.n_levels = self.n_levels.unsqueeze(1).unsqueeze(3)

        x = x.view(-1, self.size_div_batch_size, *x.size()[1:])

        self.delta = self.init_quantization_scale(x, self.dim)
        x_int = round_ste(-torch.log2(torch.clamp(x / self.delta, min=1e-8)))
        mask = x_int >= self.n_levels

        x_int = x_int / (self.n_levels - 1)
        x_quant = torch.clamp(x_int, 0, 1)
        x_quant = x_quant * (self.n_levels - 1)
        x_dequant = self.delta * 2 ** (-x_quant)
        x_dequant[mask] = 0
        x_dequant = x_dequant.squeeze().reshape(b, t, h, c).transpose(1,2)

        return x_dequant