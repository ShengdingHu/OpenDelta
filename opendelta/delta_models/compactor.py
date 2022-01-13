from functools import partial
from typing import Optional
from opendelta.delta_models.adapter import AdapterLayer
from opendelta.utils.utils import *
from opendelta.utils.cuda import get_device
from opendelta.basemodel import DeltaBase
import loralib as lora
import torch.nn as nn
import torch
import math
from opendelta.delta_models.layers.activations import Activations
import inspect
from opendelta.delta_models.layers.hypercomplex_linear import PHMLinear

class HyperComplexAdapterLayer(nn.Module):
    """Hypercomplex Adapter layer, in which the weights of up and down sampler modules
    are parameters are 1/n times of the conventional adapter layers, where n is
    hypercomplex division number."""

    def __init__(self, 
                 reduction_factor=32, 
                 non_linearity="", 
                 phm_c_init=None, 
                 hypercomplex_division=None,
                 learn_phm=True,
                 hypercomplex_nonlinearity=None,
                 shared_phm_rule=None,
                 factorized_phm=None,
                 phm_rule=None,
                 shared_W_phm=None,
                 factorized_phm_rule=None,
                 phm_rank=None,
                 phm_init_range=None,
                 kronecker_prod=None,
                 device=None, 
                 ):
        super().__init__()
        self.reduction_factor = reduction_factor
        self.non_linearity = non_linearity
        self.phm_c_init = phm_c_init
        self.hypercomplex_division = hypercomplex_division
        self.learn_phm = learn_phm
        self.phm_rule=phm_rule
        self.hypercomplex_nonlinearity = hypercomplex_nonlinearity
        self.shared_phm_rule = shared_phm_rule
        self.factorized_phm = factorized_phm
        self.shared_W_phm = shared_W_phm
        self.factorized_phm_rule = factorized_phm_rule
        self.phm_rank = phm_rank
        self.phm_init_range = phm_init_range
        self.kronecker_prod = kronecker_prod
        self.device = device

        self.instantiated = False
        

    def instantiate(self, hidden_dim):
        self.down_sample_size = hidden_dim // self.reduction_factor
        self.activation = Activations(self.non_linearity.lower()).to(self.device)
        self.down_sampler = PHMLinear(in_features=hidden_dim,
                                      out_features=self.down_sample_size,
                                      bias=True,
                                      c_init=self.phm_c_init,
                                      phm_dim=self.hypercomplex_division,
                                      phm_rule=self.phm_rule,
                                      learn_phm=self.learn_phm,
                                      w_init=self.hypercomplex_nonlinearity,
                                      shared_phm_rule=self.shared_phm_rule,
                                      factorized_phm=self.factorized_phm,
                                      shared_W_phm=self.shared_W_phm,
                                      factorized_phm_rule=self.factorized_phm_rule,
                                      phm_rank=self.phm_rank,
                                      phm_init_range=self.phm_init_range,
                                      kronecker_prod=self.kronecker_prod).to(self.device)
        self.up_sampler = PHMLinear(in_features=self.down_sample_size,
                                    out_features=hidden_dim, 
                                    bias=True,
                                    c_init=self.phm_c_init,
                                    phm_dim=self.hypercomplex_division,
                                    phm_rule=self.phm_rule,
                                    learn_phm=self.learn_phm,
                                    w_init=self.hypercomplex_nonlinearity,
                                    shared_phm_rule=self.shared_phm_rule,
                                    factorized_phm=self.factorized_phm,
                                    shared_W_phm=self.shared_W_phm,
                                    factorized_phm_rule=self.factorized_phm_rule,
                                    phm_rank=self.phm_rank,
                                    phm_init_range=self.phm_init_range,
                                    kronecker_prod=self.kronecker_prod).to(self.device)
        self.instantiated = True

    
    def forward(self, output):
        r""" Get the hidden_states from the PLM's layer output, pass it into the hypercomplex adapter, 
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



class CompactorModel(DeltaBase, nn.Module):
    r"""
    """
    def __init__(self, 
                 common_structure=False,
                 structure_mapping=None,
                 reduction_factor=16, 
                 non_linearity="gelu_new", 
                 phm_c_init="normal", 
                 hypercomplex_division=4,
                 learn_phm=True,
                 hypercomplex_nonlinearity="glorot-uniform",
                 shared_phm_rule=False,
                 factorized_phm=True,
                 shared_W_phm=False,
                 factorized_phm_rule=False,
                 phm_rank=1,
                 phm_init_range=0.0001,
                 kronecker_prod=None,
                ):
        DeltaBase.__init__(self, common_structure=common_structure, structure_mapping=structure_mapping)
        nn.Module.__init__(self)
        self.reduction_factor = reduction_factor
        self.non_linearity = non_linearity
        self.phm_c_init = phm_c_init
        self.hypercomplex_division = hypercomplex_division
        self.learn_phm = learn_phm
        self.hypercomplex_nonlinearity = hypercomplex_nonlinearity
        self.shared_phm_rule = shared_phm_rule
        self.factorized_phm = factorized_phm
        self.shared_W_phm = shared_W_phm
        self.factorized_phm_rule = factorized_phm_rule
        self.phm_rank = phm_rank
        self.phm_init_range = phm_init_range
        self.kronecker_prod = kronecker_prod

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
        adapterlayer = HyperComplexAdapterLayer(reduction_factor=16, 
                                                non_linearity=self.non_linearity, 
                                                phm_c_init=self.phm_c_init, 
                                                hypercomplex_division=self.hypercomplex_division,
                                                learn_phm=self.learn_phm,
                                                hypercomplex_nonlinearity=self.hypercomplex_nonlinearity,
                                                shared_phm_rule=self.shared_phm_rule,
                                                factorized_phm=self.factorized_phm,
                                                shared_W_phm=self.shared_W_phm,
                                                factorized_phm_rule=self.factorized_phm_rule,
                                                phm_rank=self.phm_rank,
                                                phm_init_range=self.phm_init_range,
                                                kronecker_prod=self.kronecker_prod,
                                                device=module_device
                                                )
        self.delta_modules.append(adapterlayer)  
        return adapterlayer
    