from functools import partial
from typing import Optional
from transformers.models.t5.modeling_t5 import T5Attention
from opendelta.utils.utils import *
from opendelta.utils.cuda import get_device
from opendelta.basemodel import DeltaBase
from transformers.models.t5 import T5ForConditionalGeneration # should be removed
import loralib as lora
import torch.nn as nn
import torch
import math
from opendelta.utils.modules import Activation_Function_Class





class AdapterLayer(nn.Module):
    r"""A layer of adapter tuning module. 
    """
    def __init__(self, hidden_dim, bottleneck_dim=32, non_linearity='relu', device=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim

        self.modulelist = nn.Sequential()
        self.modulelist.add_module("down_proj",nn.Linear(self.hidden_dim, self.bottleneck_dim))

        # select non-linearity
        self.modulelist.add_module("non_linear", Activation_Function_Class(non_linearity.lower()))

        self.modulelist.add_module("up_proj", nn.Linear(self.bottleneck_dim, self.hidden_dim))

        # If we want to have a layer norm on output, we apply it later after a separate residual connection
        # This means that we learn a new output layer norm, which replaces another layer norm learned in the bert layer
        # if self.add_layer_norm_after:
        #     self.adapter_norm_after = nn.LayerNorm(self.input_size)

        # if we want to initialize with the bert strategy then this function is called for all the linear layers

        
    
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
    r""" Support insert prefix token before each layer. For example, layer 3 4 6 10 and other layer untouched. 
    """
    # supported_slots = [r"encoder.*", r"decoder.*SelfAttention"]
    def __init__(self, 
                 modify_module_list=[r"encoder.*SelfAttention",r"encoder.*DenseReluDense"],
                 bottleneck_dim: Optional[int]=32, 
                 non_linearity: Optional[str]='relu',
                 sequential: Optional[str] = True
                ):
        DeltaBase.__init__(self)
        nn.Module.__init__(self)
        self.modify_module_list = modify_module_list
        self.bottleneck_dim = bottleneck_dim
        self.non_linearity = non_linearity
        self.sequential = sequential
        self.delta_modules = nn.ModuleList()
    
    def __call__(self, plm, plm_frozen=True) -> nn.Module:
        if plm_frozen:
            self.freeze_plm(plm)

        for key, _ in plm.named_modules():
            if regex_match(key, self.modify_module_list):
                _, _, ref = self.find_module(plm, key)
                print("matched_key:", key)
                adapterlayer = self.new_module_like(ref, plm.config)
                self.insert_sequential_module(ref, pre_caller=None, post_caller=adapterlayer.forward)
        return plm
    
    def new_module_like(self, module, config):
        module_device = get_device(module)
        adapterlayer = AdapterLayer(hidden_dim=config.d_model, bottleneck_dim=self.bottleneck_dim, non_linearity=self.non_linearity, device=module_device)
        self.delta_modules.append(adapterlayer)  
        return adapterlayer

    
   



        

        
    