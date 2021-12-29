
from opendelta.basemodel import DeltaBase
from opendelta.delta_models.layers.low_rank_linear import LowRankLinear
from opendelta.delta_models.layers.activations import Activations
from typing import Optional
import torch.nn as nn
import torch
from functools import partial
from typing import Optional
from opendelta.utils.utils import *
from opendelta.utils.cuda import get_device
from opendelta.basemodel import DeltaBase
import loralib as lora
import torch.nn as nn
import torch
import math


class LowRankAdapterLayer(nn.Module):
    """This is the low-rank adapter, in which each adapter is composed of two rank-one matrices.
    """
    def __init__(self, 
                 reduction_factor=32, 
                 non_linearity="gelu_new",
                 low_rank_w_init="glorot-uniform", 
                 low_rank_rank=1,
                 device=None):
        super().__init__()
        self.reduction_factor = reduction_factor
        self.non_linearity = non_linearity
        self.low_rank_w_init = low_rank_w_init
        self.low_rank_rank = low_rank_rank
        self.device = device
        self.instantiated = False

    
    def instantiate(self, hidden_dim):

        self.down_sample_size = hidden_dim // self.reduction_factor
        self.activation = Activations(self.non_linearity.lower()).to(self.device)
        self.down_sampler = LowRankLinear(hidden_dim, self.down_sample_size,
                                          w_init=self.low_rank_w_init,
                                          rank=self.low_rank_rank).to(self.device)
        self.up_sampler = LowRankLinear(self.down_sample_size, hidden_dim,
                                        w_init=self.low_rank_w_init,
                                        rank=self.low_rank_rank).to(self.device)

        self.instantiated = True

    def forward(self, output):
        r""" Get the hidden_states from the PLM's layer output, pass it into the low-rank adapter, 
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
            

        z = self.down_sampler(hiddens)
        z = self.activation(z)
        adapter_output = self.up_sampler(z)

        modified_output = adapter_output + hiddens # residual_connection
        if isinstance(output, tuple):
            output = (modified_output,) + output[1:]
        elif isinstance(output, torch.Tensor):
            output = modified_output
        else:
            raise TypeError
        return output






class LowRankAdapterModel(DeltaBase, nn.Module):
    r"""
    """
    def __init__(self, 
                 common_structure = False,
                 structure_mapping = None,
                 reduction_factor = 32,
                 non_linearity = "gelu_new",
                 low_rank_w_init = "glorot-uniform", 
                 low_rank_rank = 1,
                 ):
            
        DeltaBase.__init__(self, common_structure=common_structure, structure_mapping=structure_mapping)
        nn.Module.__init__(self)
        self.reduction_factor = reduction_factor
        self.non_linearity = non_linearity
        self.low_rank_w_init = low_rank_w_init
        self.low_rank_rank = low_rank_rank
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
        self.pseudo_data_to_instantiate(module)
        setattr(module, registration_name, self)
        self.mark_as_delta()
        return module
    
    def update_module(self, module: nn.Module, key: str):
        _, _, ref = self.find_module(module, key)
        adapterlayer = self.new_module_like(ref)
        self.insert_sequential_module(ref, pre_caller=None, post_caller=adapterlayer.forward)
    
    def new_module_like(self, module):
        module_device = get_device(module)
        adapterlayer = LowRankAdapterLayer(reduction_factor = self.reduction_factor,
                                      non_linearity = self.non_linearity,
                                      low_rank_w_init = self.low_rank_w_init, 
                                      low_rank_rank = self.low_rank_rank,
                                      device=module_device)
        self.delta_modules.append(adapterlayer)  
        return adapterlayer
    