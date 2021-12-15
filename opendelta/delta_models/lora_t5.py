from typing import Optional
from opendelta.utils.utils import *
from opendelta.basemodel import DeltaBase
from transformers.models.t5 import T5ForConditionalGeneration
import loralib as lora
import torch.nn as nn


class LoraModel(DeltaBase, nn.Module):
    def __init__(self, 
                 modify_module_list=["SelfAttention.q","SelfAttention.v"],
                 lora_r=8,
                 lora_alpha=16,
                 lora_dropout=0.0,):
        DeltaBase.__init__(self)
        nn.Module.__init__(self)
        self.modify_module_list = modify_module_list
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_modules = nn.ModuleList()
    
    def __call__(self, plm, plm_frozen=True) -> nn.Module:
        if plm_frozen:
            self.freeze_plm(plm)
        for key, _ in plm.named_modules():
            if substring_in(key, self.modify_module_list):
                print("key",key)
                parent_ref, children_name, child_ref = self.find_module(plm, key)
                self.replace_module(parent_ref, children_name, child_ref)
        return plm
    
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
            self.lora_modules.append(new_module)
        else:
            raise NotImplementedError

        setattr(parent_module, children_name, new_module)