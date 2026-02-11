import torch.nn as nn
from utils.models import MatMul
from quant.quant_block import subblock_specials, block_specials, BaseQuantBlock
from quant.quant_layer import QuantModule, QuantMatMul

class QuantModel(nn.Module):
    def __init__(self, model: nn.Module, weight_quant_params: dict = {}, act_quant_params: dict = {}, matmul_params: dict = {}, n_groups: int = 1, calib_granularity: str = 'block'):
        super().__init__()
        self.model = model
        self.calib_granularity = calib_granularity
        self.quant_module_refactor(self.model, weight_quant_params, act_quant_params, matmul_params, n_groups)

    def quant_module_refactor(self, module: nn.Module, weight_quant_params: dict = {}, act_quant_params: dict = {}, matmul_params: dict = {}, n_groups: int = 1):
        """
        Recursively replace the normal conv2d and Linear layer to QuantModule
        :param module: nn.Module with nn.Conv2d or nn.Linear in its children
        :param weight_quant_params: quantization parameters like n_bits for weight quantizer
        :param act_quant_params: quantization parameters like n_bits for activation quantizer
        """
        specials = subblock_specials if self.calib_granularity == 'subblock' else block_specials
        
        for name, child_module in module.named_children():
            if type(child_module) in specials:
                setattr(module, name, specials[type(child_module)](child_module, weight_quant_params, act_quant_params, matmul_params, n_groups))

            elif isinstance(child_module, (nn.Conv2d, nn.Linear)):
                setattr(module, name, QuantModule(child_module, weight_quant_params, act_quant_params, n_groups))

            elif isinstance(child_module, MatMul):
                setattr(module, name, QuantMatMul(matmul_params, n_groups, positive = child_module.positive))

            else:
                self.quant_module_refactor(child_module, weight_quant_params, act_quant_params, matmul_params, n_groups)

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        for m in self.model.modules():
            if isinstance(m, (QuantModule, BaseQuantBlock)):
                m.set_quant_state(weight_quant, act_quant)

            elif isinstance(m, QuantMatMul):
                m.set_quant_state(act_quant, act_quant)

    def forward(self, input):
        return self.model(input)

    def set_first_last_layer_to_8bit(self):
        module_list = []
        for m in self.model.modules():
            if isinstance(m, (QuantModule, QuantMatMul)):
                module_list += [m]
        module_list[0].weight_quantizer.bitwidth_refactor(8)
        module_list[-1].weight_quantizer.bitwidth_refactor(8)
        module_list[-1].act_quantizer.bitwidth_refactor(8)