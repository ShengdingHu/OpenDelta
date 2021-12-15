from opendelta.utils.utils import *
from opendelta.basemodel import DeltaBase
from transformers.models.t5 import T5ForConditionalGeneration
import loralib as lora
import torch.nn as nn

from transformers.models.bert.modeling_bert import BertForMaskedLM
import torch
from torch.nn import init
import math

class BiasModel(DeltaBase, nn.Module):
    supported_slots = ["SelfAttention", "DenseReluDense", "layer_norm"]
    def __init__(self, 
                 modify_module_list=["SelfAttention", "DenseReluDense"]):
        DeltaBase.__init__(self)
        nn.Module.__init__(self)
        self.modify_module_list = [x for x in modify_module_list if x in self.supported_slots]
        self.delta_params = nn.ParameterList()
    
    def __call__(self, plm, plm_frozen=True) -> nn.Module:
        if plm_frozen:
            self.freeze_plm(plm)
        for key, _ in plm.named_modules():
            if substring_in(key, self.modify_module_list):
                print("key",key)
                _, _, ref = self.find_module(plm, key)
                self.modify_module(ref)
        return plm

    def modify_module(self,
                      module: nn.Module, 
                      ):
        if isinstance(module, nn.Linear):
            if module.bias is None:
                bias = nn.Parameter(torch.empty(module.out_features), requires_grad=True)
                module.register_parameter('bias', bias)
                self._reset_bias_parameters(module)
                self.delta_params.append(bias)
            else:
                self.delta_params.append(module.bias)
        else:
            pass
    
    @staticmethod
    def _reset_bias_parameters(linear_module):
        fan_in, _ = init._calculate_fan_in_and_fan_out(linear_module.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        init.uniform_(linear_module.bias, -bound, bound)