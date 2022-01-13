from functools import partial
from typing import Optional
from opendelta.utils.utils import *
from opendelta.utils.cuda import get_device
from opendelta.basemodel import DeltaBase
import loralib as lora
import torch.nn as nn
import torch
import math
from opendelta.delta_models.layers.activations import Activations
import inspect





class AdapterLayer(nn.Module):
    r"""A layer of adapter tuning module. 
    """
    def __init__(self, bottleneck_dim=32, non_linearity='relu', device=None):
        super().__init__()
        self.bottleneck_dim = bottleneck_dim
        self.device = device
        self.instantiated = False
        self.non_linearity = non_linearity
    
    def instantiate(self, hidden_dim):
        self.modulelist = nn.Sequential()
        self.modulelist.add_module("down_proj",nn.Linear(hidden_dim, self.bottleneck_dim, device=self.device))

        # select non-linearity
        self.modulelist.add_module("non_linear", Activations(self.non_linearity.lower()))

        self.modulelist.add_module("up_proj", nn.Linear(self.bottleneck_dim, self.hidden_dim,  device=self.device))

        # TODO:
        # If we want to have a layer norm on output, we apply it later after a separate residual connection
        # This means that we learn a new output layer norm, which replaces another layer norm learned in the bert layer
        # if self.add_layer_norm_after:
        #     self.adapter_norm_after = nn.LayerNorm(self.input_size)

        # if we want to initialize with the bert strategy then this function is called for all the linear layers
        self.instantiated = True
        
    
    def forward(self, output):
        r""" Get the hidden_states from the PLM's layer output, pass it into the adapter, 
        then combined with the main hidden_states. Finally pass it into the subsequent layer.

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
            

        adapter_output = self.modulelist(hiddens)
        modified_output = adapter_output + hiddens # residual_connection
        if isinstance(output, tuple):
            output = (modified_output,) + output[1:]
        elif isinstance(output, torch.Tensor):
            output = modified_output
        else:
            raise TypeError
        return output
    
  

class AdapterModel(DeltaBase, nn.Module):
    r"""
    """
    def __init__(self, 
                 bottleneck_dim: Optional[int]=32, 
                 non_linearity: Optional[str]='relu',
                 sequential: Optional[str] = True,
                 common_structure = False,
                 structure_mapping = None,
                ):
        DeltaBase.__init__(self, common_structure=common_structure, structure_mapping=structure_mapping)
        nn.Module.__init__(self)
        self.bottleneck_dim = bottleneck_dim
        self.non_linearity = non_linearity
        self.sequential = sequential
        self.delta_modules = nn.ModuleList()
        
    def __call__(self, 
                 module: nn.Module, 
                 modified_keys: List[str],
                 is_regex: Optional[bool]=False,
                 registration_name: Optional[str] = "deltas"
                ) -> nn.Module:
        for key, _ in module.named_modules():
            if self.find_key(key, modified_keys, is_regex):
                # print("find key",key)
                self.update_module(module, key)
        self._pseudo_data_to_instantiate(module)
        setattr(module, registration_name, self)
        self.mark_as_delta()
        return module
    
    def update_module(self, module: nn.Module, key: str):
        _, _, ref = self.find_module(module, key)
        adapterlayer = self.new_module_like(ref)
        self.insert_sequential_module(ref, pre_caller=None, post_caller=adapterlayer.forward)
    
    def new_module_like(self, module):
        module_device = get_device(module)
        adapterlayer = AdapterLayer(bottleneck_dim=self.bottleneck_dim, non_linearity=self.non_linearity, device=module_device)
        self.delta_modules.append(adapterlayer)  
        return adapterlayer
    