from typing import Optional
from opendelta.utils.signature import get_arg_names_inside_func
from opendelta.utils.utils import *
from opendelta.basemodel import DeltaBase, is_leaf_module
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

class BiasLayer(nn.Module):
    def __init__(self, init_method="zero"):
        super().__init__()
        self.init_method=init_method
        self.instantiated = False

    def instantiate(self, hidden_dim):
        if self.init_method == "zero":
            self.bias = nn.Parameter(torch.zeros(hidden_dim))
        else:
            raise NotImplementedError
        self.instantiated = True
    
    def forward(self, output):
        r"""Presuming the first argument is the tensor to add bias along the last dimension.
        In most cases, it is correct. However, be aware of the possibility that the presumption
        doesn't hold. 
        """
        if isinstance(output, tuple):
            hiddens = output[0]
        elif isinstance(output, torch.Tensor):
            hiddens = output
        else:
            raise TypeError
        
        if not self.instantiated:
            self.hidden_dim = hiddens.shape[-1]
            print(f"Got hidden dim hidden_dim {self.hidden_dim}")
            self.instantiate(hidden_dim=self.hidden_dim)

        modified_output = hiddens + self.bias
        
        if isinstance(output, tuple):
            output = (modified_output,) + output[1:]
        elif isinstance(output, torch.Tensor):
            output = modified_output
        else:
            raise TypeError
        return output



class BitFitModel(DeltaBase, nn.Module):


    config_class = BitFitConfig
    delta_type = "bitfit"
    default_modified_modules = ["attn", "ff", "layer_norm","lm_head.proj"] # modify all the bias parameter in attention and feed-forward layer.
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
        self.delta_modules = nn.ModuleList()

        self.add_all_delta_to_backbone(self.backbone_model,
                                   self.modified_modules,
                                   self.registration_name)
    
    
    def update_module(self, module: nn.Module, key: str):
        _, _, ref = self.find_module(module, key)
        self.modify_module(ref)
        

    def modify_module(self,
                      module: nn.Module, 
                      ):
        if is_leaf_module(module):
            # if it is a leaf module, add bias to it regardless of its type.
            if isinstance(module, nn.Linear):
                self.add_bias_to_linear(module)
            else:
                # for example, layer_norms, lm_heads.
                self.add_bias_to_others(module)
        else:
            # for the non-leaf modules, by default it will add bias only to the linear submodules.
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
   
    def add_bias_to_linear(self, c):
        if c.bias is None:
            bias = nn.Parameter(torch.empty(c.out_features), requires_grad=True)
            c.register_parameter('bias', bias)
            self._reset_bias_parameters(c)
            self.delta_params.append(bias)
        else:
            c.bias.requires_grad = True
            self.delta_params.append(c.bias)
    
    def add_bias_to_others(self, c):
        new_bias = BiasLayer()
        self.insert_sequential_module(c, pre_caller=None, post_caller=new_bias.forward, delta_module=new_bias, name="bias")
        self.delta_modules.append(new_bias)
    # def _pseudo_data_to_instantiate(self, module):
    #     # no need to pass pseudo input
        

    
    @staticmethod
    def _reset_bias_parameters(linear_module):
        fan_in, _ = init._calculate_fan_in_and_fan_out(linear_module.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        init.uniform_(linear_module.bias, -bound, bound)
    
    def register_delta_if_new(self, module: nn.Module, registration_name: Optional[str] = "deltas"):
        # The BiasModel is not new to the backbone model. 
        return