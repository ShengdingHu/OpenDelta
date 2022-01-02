from opendelta.utils.utils import *
from opendelta.basemodel import DeltaBase
from transformers.models.t5 import T5ForConditionalGeneration
import loralib as lora
import torch.nn as nn

from transformers.models.bert.modeling_bert import BertForMaskedLM
import torch
from torch.nn import init
import math
from opendelta.utils.structure_mapping import transform

class BiasModel(DeltaBase, nn.Module):
    def __init__(self, 
                 modify_module_list=["attn", "ff"], 
                 common_structure=False,
                 structure_mapping=None):
        DeltaBase.__init__(self,common_structure=common_structure, structure_mapping=structure_mapping )
        nn.Module.__init__(self)
        self.delta_params = nn.ParameterList()
    
    
    def update_module(self, module: nn.Module, key: str):
        _, _, ref = self.find_module(module, key)
        self.modify_module(ref)
        

    def modify_module(self,
                      module: nn.Module, 
                      ):
        for n, c in module.named_modules():
            if isinstance(c, nn.Linear):
                if c.bias is None:
                    bias = nn.Parameter(torch.empty(c.out_features), requires_grad=True)
                    c.register_parameter('bias', bias)
                    self._reset_bias_parameters(c)
                    self.delta_params.append(bias)
                else:
                    c.bias.requires_grad = True
                    self.delta_params.append(c.bias)
            else:
                pass
    
    @staticmethod
    def _reset_bias_parameters(linear_module):
        fan_in, _ = init._calculate_fan_in_and_fan_out(linear_module.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        init.uniform_(linear_module.bias, -bound, bound)
    
    def register_delta_if_new(self, module: nn.Module, registration_name: Optional[str] = "deltas"):
        # Do nothing since lora modules is a part of the original module.
        pass