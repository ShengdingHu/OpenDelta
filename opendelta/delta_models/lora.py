from typing import Optional
from opendelta.utils.utils import *
from opendelta.basemodel import DeltaBase
from transformers.models.t5 import T5ForConditionalGeneration
import loralib as lora
import torch.nn as nn

class LoraModel(DeltaBase, nn.Module):
    def __init__(self,
                 lora_r=8,
                 lora_alpha=16,
                 lora_dropout=0.0,
                 common_structure=False,
                 structure_mapping=None
                 ):
        DeltaBase.__init__(self, common_structure=common_structure, structure_mapping=structure_mapping)
        nn.Module.__init__(self)
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.delta_modules = nn.ModuleList()
    
    
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
            new_module.bias = child_module.bias
            self.delta_modules.append(new_module)
        else:
            from IPython import embed
            print("line49")
            embed()
            raise NotImplementedError

        setattr(parent_module, children_name, new_module)
    
    def mark_as_delta(self, module: nn.Module = None):
        if module is None:
            module=self
        for n, p in module.named_parameters():
            para_name = n.split(".")[-1]
            if "weight" not in para_name: # only lora_A, lora_B is the delta parameter.
                setattr(p, "_is_delta", True)