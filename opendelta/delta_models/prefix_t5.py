from opendelta.utils.utils import *
from opendelta.basemodel import DeltaBase
from transformers.models.t5 import T5ForConditionalGeneration
import loralib as lora
import torch.nn as nn


class PrefixModel(DeltaBase, nn.Module):
    r"Not written yet."
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
    
    def __call__(self, plm) -> nn.Module:
        for key, _ in plm.named_modules():
            if substring_in(key, self.modify_module_list):
                print("key",key)
                self.modify_module(plm, key)
        return plm


    def new_module_like(self, module):
        if isinstance(module, nn.Linear):
            in_features, out_features = module.in_features, module.out_features
            new_module = lora.Linear(in_features=in_features, 
                                     out_features=out_features, 
                                     r=self.lora_r, 
                                     lora_alpha=self.lora_alpha,
                                     lora_dropout=self.lora_dropout)
            self.lora_modules.append(new_module)
        else:
            raise NotImplementedError
        return new_module
    
    
    
    




        






def get_lora_t5(t5model):
    loramodel = LoraModel(modify_module_list = ["SelfAttention.q","SelfAttention.v"])
    print([name for name, v in t5model.named_modules()])
    t5model = loramodel(t5model)
    print([name for name, v in t5model.named_modules()])
    exit()
    return t5model

