from typing import Optional
from opendelta.utils.signature import get_arg_names_inside_func
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
from opendelta import BaseDeltaConfig



class BitFitConfig(BaseDeltaConfig):
    r"""
    This is the configuration class to store the configuration of a [`LoraModel`]

    """
    def __init__(
        self, 
        **kwargs
    ): 
        super().__init__(**kwargs)
        arg_names = get_arg_names_inside_func(self.__init__)
        for arg_name in arg_names:
            if not hasattr(self, arg_name): # the arg has not been registered in parent config
                setattr(self, arg_name, locals()[arg_name])



class BitFitModel(DeltaBase, nn.Module):


    config_class = BitFitConfig
    delta_type = "bitfit"
    default_modified_modules = ["attn", "ff"] # modify all the bias parameter in attention and feed-forward layer.
    def __init__(self,
                 backbone_model: nn.Module, 
                 modified_modules: Optional[bool] = None,
                 unfrozen_modules: Optional[bool] = None,
                 common_structure: Optional[bool] = None,
                 registration_name: Optional[str] = "deltas",
                 ):
        DeltaBase.__init__(self, 
                           backbone_model, 
                           modified_modules=modified_modules,
                           unfrozen_modules=unfrozen_modules,
                           common_structure=common_structure,
                           registration_name=registration_name
                           )
        arg_names = get_arg_names_inside_func(self.__init__)
        for arg_name in arg_names:
            if not hasattr(self, arg_name): # not registered in parent class
                setattr(self, arg_name, locals()[arg_name])

        self.delta_params = nn.ParameterList()

        self.add_all_delta_to_backbone(self.backbone_model,
                                   self.modified_modules,
                                   self.registration_name)
    
    
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
        # The BiasModel is not new to the backbone model. 
        return