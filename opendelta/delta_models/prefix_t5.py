from functools import partial
from typing import Optional
from transformers.models.t5.modeling_t5 import T5Attention
from opendelta.utils.utils import *
from opendelta.utils.cuda import get_device
from opendelta.basemodel import DeltaBase
from transformers.models.t5 import T5ForConditionalGeneration
import loralib as lora
import torch.nn as nn
import torch


class PrefixLayer(nn.Module):
    r"""A layer of prefix tuning module. The layer's forward function pass (or concatenate) the additional past_key_value
    into the original attention layer's forward function.
    """
    def __init__(self, prefix_token_num, num_heads, d_kv,  device,):
        super().__init__()
        self.num_heads = num_heads
        self.d_kv = d_kv
        self.shape1 = num_heads*d_kv
        self.prefix_token_num = prefix_token_num
        self.past_key = nn.Parameter(torch.randn(prefix_token_num, num_heads * d_kv, device=device), requires_grad=True)
        self.past_value = nn.Parameter(torch.randn(prefix_token_num, num_heads * d_kv, device=device), requires_grad=True)
        self.past_key_reparam = self.past_key.data
        self.past_value_reparam = self.past_value.data
        
    
    def forward(self, *args, **kwargs):
        r"""The args and kwargs are inherited from the T5Attention's forward function.
        """
        batch_size = args[0].shape[0]
 
        def expand_batchsize(x):
            x = x.reshape(self.prefix_token_num, self.num_heads, self.d_kv).transpose(0,1)
            x = x.unsqueeze(0).expand(batch_size, *x.shape)
            return x
        # from IPython import embed
        # embed()
        if 'position_bias' in kwargs and kwargs['position_bias'] is not None:
            print("positional bias not None")
            if kwargs['position_bias'].shape[-1] != args[0].shape[-2] + self.prefix_token_num: # Then the position_bias should be re-calculated 
                kwargs['position_bias'] = None
        if kwargs['past_key_value'] is None:
            kwargs['past_key_value'] = (expand_batchsize(self.past_key_reparam), expand_batchsize(self.past_value_reparam))
        if 'mask' in kwargs and kwargs['mask'] is not None:
            am = kwargs['mask']  # Should check the format of the attention_mask when moving to a new plm.
            kwargs['mask'] = torch.cat([-torch.zeros((*am.shape[:-1],self.prefix_token_num), dtype = am.dtype,device=am.device), am], dim=-1)
        return args, kwargs
    
    def post_forward(self, output):
        r""" Remove the cached positional bias, since the next layer may not have prefix token. 
        """
        output = output[:2] + (None, )+ output[3:]
        return output
    
    
    

class ReparameterizeFunction(nn.Module):
    r""" Prefix Tuning's performance is better with a reparameterize module, which generates
    the `past_key_value` using an MLP instead of directly optimizing the `past_key_value` as leaf variable.
    In our implementation, the reparameterize module is constructed according to the number of parameters 
    in all `past_key_value`s. Thus, variable number of prefixlayer is supported (not restricting to being equal
    to the number of layers of the pretraind language model)


    """
    def __init__(self, prefix_token_num, d_model,  dropout_rate=0.0, mid_dim=512, module_list=[]):
        super().__init__()
        self.prefix_token_num = prefix_token_num
        self.d_model = d_model
        self.mid_dim = mid_dim
        self.module_list = module_list
        self.dropout = nn.Dropout(dropout_rate)
        self.record_parameters()
        self.compatibility_check()
        self.define_reparameterization_network()
        
    def record_parameters(self):
        r""" Enumerate the parameters that need to be reparameterized.
        Then, delete the original parameters. 
        """
        tot = 0
        for module in self.module_list:
            for n, parameters in module.named_parameters():
                tot += parameters.numel()
                module.register_parameter(n, None)
        self.total_parameters_num = tot
    
    def compatibility_check(self,):
        r"""May be removed.
        """
        assert self.total_parameters_num % self.prefix_token_num == 0
    
    def allocate_parameter(self):
        r""" At the beginning of each forward pass through the whole network(PLM), 
        cacalulate the reparameterized past_key and past_value (`past_key_reparam` and `past_value_reparam`)
        for later use in each layer.
        """
        input_tokens = self.input_tokens
        temp_control = self.wte(input_tokens)
        past_key_values = self.control_trans(temp_control)
        seqlen, _ = past_key_values.shape

        past_key_values = past_key_values.view(seqlen, len(self.module_list) * 2, self.module_list[0].shape1)
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([1, 0, 2]).split(2)

        for module_id, module in enumerate(self.module_list):
            module.past_key_reparam = past_key_values[module_id][0]
            module.past_value_reparam = past_key_values[module_id][1]

    def forward(self, *args, **kwargs):
        r""" Firstly forward through the reparameterized network, and then go through normal forward pass of the PLM.
        """
        self.allocate_parameter()
        return args, kwargs

    def define_reparameterization_network(self) -> None:
        r""" Build the reparameterize module 
        """
        self.input_tokens = nn.Parameter(torch.arange(self.prefix_token_num).long(), requires_grad=False) # to allow automatic devicing
        self.wte = nn.Embedding(self.prefix_token_num, self.d_model)
        self.control_trans = nn.Sequential(
            nn.Linear(self.d_model, self.mid_dim),
            nn.Tanh(),
            nn.Linear(self.mid_dim, self.total_parameters_num//self.prefix_token_num)
        )




class PrefixModel(DeltaBase, nn.Module):
    r""" Support insert prefix token before each layer. For example, layer 3 4 6 10 and other layer untouched. 
    """
    supported_slots = [r"encoder.*SelfAttention", r"decoder.*SelfAttention"]
    def __init__(self, 
                 modify_module_list=[r"encoder.*SelfAttention"],
                 prefix_token_num=6,
                 reparameterize=True,
                 mid_dim: Optional[int]=512
                ):
        DeltaBase.__init__(self)
        nn.Module.__init__(self)
        self.modify_module_list = modify_module_list
        self.prefix_token_num = prefix_token_num
        self.reparameterize = reparameterize
        self.mid_dim = mid_dim
        self.delta_modules = nn.ModuleList()
    
    def __call__(self, plm, plm_frozen=True) -> nn.Module:
        if plm_frozen:
            self.freeze_plm(plm)

        for key, _ in plm.named_modules():
            if regex_match(key, self.modify_module_list):
                _, _, ref = self.find_module(plm, key)
                print("matched_key:", key)
                prefixlayer = self.new_module_like(ref, plm.config)
                self.insert_sequential_module(ref, pre_caller=prefixlayer.forward, post_caller=prefixlayer.post_forward)
        
        if self.reparameterize:
            reparams = ReparameterizeFunction(prefix_token_num=self.prefix_token_num, d_model=plm.config.d_model, module_list=self.delta_modules)
            self.delta_modules = None
            self.reparams = reparams
            self.insert_sequential_module(plm, pre_caller=reparams.forward)

        return plm
    
    def new_module_like(self, module, config):
        if isinstance(module, T5Attention):
            module_device = get_device(module)
            prefixlayer = PrefixLayer(prefix_token_num=self.prefix_token_num, num_heads=config.num_heads, d_kv=config.d_kv, device=module_device)
            self.delta_modules.append(prefixlayer)  
            return prefixlayer
        else:
            raise NotImplementedError
    
   



        

        
    