from typing import Optional

from opendelta.utils.signature import get_arg_names, get_arg_names_inside_func
from opendelta.utils.utils import *
from opendelta.basemodel import DeltaBase
from transformers.models.t5 import T5ForConditionalGeneration
import loralib as lora
import torch.nn as nn
from opendelta import BaseDeltaConfig

class LoraConfig(BaseDeltaConfig):
    r"""
    This is the configuration class to store the configuration of a [`LoraModel`]

    """
    def __init__(
        self, 
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        **kwargs
    ): 
        super().__init__(**kwargs)
        arg_names = get_arg_names_inside_func(self.__init__)
        for arg_name in arg_names:
            if not hasattr(self, arg_name): # the arg has not been registered in parent config
                setattr(self, arg_name, locals()[arg_name])



class LoraModel(DeltaBase):

    config_class = LoraConfig
    delta_type = "lora"
    default_modified_modules = ['attn.q', 'attn.v']
    def __init__(self,
                 backbone_model: nn.Module, 
                 lora_r=8,
                 lora_alpha=16,
                 lora_dropout=0.0,
                 modified_modules: Optional[bool] = None,
                 common_structure: Optional[bool] = None,
                 registration_name: Optional[str] = "deltas",
                 ):
        DeltaBase.__init__(self, 
                           backbone_model, 
                           modified_modules=modified_modules,
                           common_structure=common_structure,
                           registration_name=registration_name
                           )
        arg_names = get_arg_names_inside_func(self.__init__)
        from IPython import embed
        embed(header= "in lora")
        for arg_name in arg_names:
            if not hasattr(self, arg_name): # not registered in parent class
                setattr(self, arg_name, locals()[arg_name])

        self.delta_modules = nn.ModuleList()

        self.add_all_delta_to_backbone(backbone_model,
                                   modified_modules,
                                   registration_name)
    
    # @classmethod
    # def from_config(cls, config: LoraConfig):

    
    
    def update_module(self, module: nn.Module, key: str):
        parent_ref, children_name, child_ref = self.find_module(module, key)
        self.replace_module(parent_ref, children_name, child_ref)
        
    
    def replace_module(self,
                      parent_module: nn.Module, 
                      children_name: str,
                      child_module: Optional[nn.Module],
                      ):
        r"""Replace a module's child module (the entire referenced module) using the module's reference name.
        If the replacemodule is None, it will call self.new_module_like method which are
        different for different objects. 
        This method will get the reference of the parent modules of the target module and register a new module on the node. 
        """
        if isinstance(child_module, nn.Linear):
            in_features, out_features = child_module.in_features, child_module.out_features
            new_module = lora.Linear(in_features=in_features, 
                                     out_features=out_features, 
                                     r=self.lora_r, 
                                     lora_alpha=self.lora_alpha,
                                     lora_dropout=self.lora_dropout)
            new_module.weight = child_module.weight
            new_module.bias = child_module.bias # if bias is None, also copy
            self.delta_modules.append(new_module)
        else:
            raise NotImplementedError

        setattr(parent_module, children_name, new_module)
    
    def mark_as_delta(self, module: nn.Module = None):
        if module is None:
            module=self
        for n, p in module.named_parameters():
            param_name = n.split(".")[-1]
            if "lora_A" in param_name or "lora_B" in param_name: # only lora_A, lora_B is the delta parameter.
                setattr(p, "_is_delta", True)
    
    def register_delta_if_new(self, module: nn.Module, registration_name: Optional[str] = "deltas"):
        # Do nothing since lora modules is a part of the original module.
        pass
        