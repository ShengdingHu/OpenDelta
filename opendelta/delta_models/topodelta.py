
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
from opendelta.delta_models.layers.init import glorot_uniform, glorot_normal
import numpy as np
from tensorboardX import SummaryWriter
writer = SummaryWriter()
global glb_iter
glb_iter = 0

class DegradedLinear(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int, rank: int = 1,
        bias: bool = True, w_init: str = "glorot-uniform"):
        super(DegradedLinear, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim 
        self.rank = rank
        self.bias = bias
        self.w_init = w_init
        self.total_param = input_dim * output_dim
        self.real_param = input_dim * rank + output_dim * rank


        self.W_base = nn.Parameter(torch.Tensor(size=(self.real_param,)), requires_grad=True)

        self.init_range = math.sqrt(1 / (input_dim + input_dim))

        # self.W_0 = nn.Parameter(torch.Tensor(size=(input_dim, rank)), requires_grad=True)
        # self.W_1 = nn.Parameter(torch.Tensor(size=(rank, output_dim)), requires_grad=True)
        if bias:
            self.b = nn.Parameter(torch.Tensor(output_dim))
        self.reset_parameters()
    
    def reset_parameters(self):
        if self.bias:
            self.b.data = torch.zeros_like(self.b.data)
        if self.w_init == "glorot-uniform": 
            self.W_base.data = torch.nn.init.uniform_(self.W_base, a=-math.sqrt(6)*self.init_range, b=math.sqrt(6)*self.init_range) 
            # self.W_right.data = uniform(self.W_right.data)          
        elif self.w_init == "glorot-normal":
            self.W_base.data = torch.nn.init.normal_(self.W_base, mean=0, std=math.sqrt(2)*self.init_range)
            # self.W_right.data = normal(self.W_right.data)
        else:
            raise ValueError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # from IPython import embed
        # embed()
        W = torch.cat([self.W_base.repeat(self.total_param//self.real_param), self.W_base[:self.total_param - self.total_param//self.real_param*self.real_param]], dim=0)
        W = W.reshape(self.input_dim, self.output_dim)
        
        # from IPython import embed
        # embed()
        # W = self.W_left*self.W_right
        output = torch.matmul(input=x, other=W)
        if self.bias:
            output += self.b
        return output

class RemappingLinear(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int, rank: int = 1,
        bias: bool = True, w_init: str = "glorot-uniform"):
        super(RemappingLinear, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim 
        self.rank = rank
        self.bias = bias
        self.w_init = w_init
        self.total_param = input_dim * output_dim
        # self.real_param = input_dim * rank + output_dim * rank
        self.random_mapping = np.arange(0, self.total_param)
        np.random.shuffle(self.random_mapping)
        self.random_mapping = torch.from_numpy(self.random_mapping)
        

        # self.W_base = nn.Parameter(torch.Tensor(size=(self.real_param,)), requires_grad=True)

        self.w_init = w_init
        self.W_left = nn.Parameter(torch.Tensor(size=(input_dim, rank)), requires_grad=True)
        self.W_right = nn.Parameter(torch.Tensor(size=(rank, output_dim)), requires_grad=True)
        if bias:
            self.b = nn.Parameter(torch.Tensor(output_dim))
        self.reset_parameters()
    
    def reset_parameters(self):
        if self.bias:
            self.b.data = torch.zeros_like(self.b.data)
        if self.w_init == "glorot-uniform": 
            self.W_left.data = glorot_uniform(self.W_left.data) 
            self.W_right.data = glorot_uniform(self.W_right.data)          
        elif self.w_init == "glorot-normal":
            self.W_left.data = glorot_normal(self.W_left.data)
            self.W_right.data = glorot_normal(self.W_right.data)
        else:
            raise ValueError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # from IPython import embed
        # embed()
        W = self.W_left*self.W_right
        W = W.reshape(self.total_param)
        W = W[self.random_mapping]

        # W = torch.cat([self.W_base.repeat(self.total_param//self.real_param), self.W_base[:self.total_param - self.total_param//self.real_param*self.real_param]], dim=0)
        W = W.reshape(self.input_dim, self.output_dim)
        
        # from IPython import embed
        # embed()
        # W = self.W_left*self.W_right
        output = torch.matmul(input=x, other=W)
        if self.bias:
            output += self.b
        return output

class HierarchicalLowRankSubLayer(torch.nn.Module):
    def __init__(self, input_dim=None, output_dim=None, target_dim: int=None, rank: int = 1, recursion_depth=0, target_depth=1, is_left=True,
        bias: bool = False, w_init: str = "glorot-uniform"):
        super(HierarchicalLowRankSubLayer, self).__init__()
        self.glb_iter = 0
        self.recursion_depth = recursion_depth
        self.input_dim = input_dim
        self.output_dim = output_dim
        if target_dim == None:
            self.target_dim = self.input_dim * self.output_dim
        else:
            self.target_dim = target_dim
        
        self.target_depth = target_depth
        self.is_left = is_left
        
        self.source_dim = int(math.sqrt(self.target_dim)//1+1)
        # self.source_dim1 = self.input_dim
        # self.source_dim2 = self.output_dim



        self.rank = rank
        self.bias = bias

        self.w_init = w_init
        if self.recursion_depth == self.target_depth:
            # try:
            self.W_left = nn.Parameter(torch.Tensor(size=(self.source_dim, rank)), requires_grad=True)
            self.W_right = nn.Parameter(torch.Tensor(size=(rank, self.source_dim)), requires_grad=True)
            # self.W_left.register_hook(lambda grad: print(grad))
            # except:
            #     from IPython import embed
            #     embed()
            self.initialize_parameters()
        elif self.recursion_depth < self.target_depth:
            self.W_left = HierarchicalLowRankSubLayer(target_dim=self.source_dim, 
                            rank=rank*2, 
                            recursion_depth=self.recursion_depth+1, 
                            target_depth=self.target_depth, is_left=True)
            self.W_right = HierarchicalLowRankSubLayer(target_dim=self.source_dim, 
                            rank=rank*2, 
                            recursion_depth=self.recursion_depth+1, 
                            target_depth=self.target_depth, is_left=False)
            
        else:
            raise RuntimeError
        
        if self.recursion_depth==0:
            if self.bias:
                self.b = nn.Parameter(torch.Tensor(self.output_dim))
                self.b.data = torch.zeros_like(self.b.data)
        
    def initialize_parameters(self,):
        if self.recursion_depth == self.target_depth:
            if self.w_init == "glorot-uniform": 
                self.W_left.data = glorot_uniform(self.W_left.data) 
                self.W_right.data = glorot_uniform(self.W_right.data)          
            elif self.w_init == "glorot-normal":
                self.W_left.data = glorot_normal(self.W_left.data)
                self.W_right.data = glorot_normal(self.W_right.data)
            else:
                raise ValueError

    def generate_parameters(self):
        if self.recursion_depth == self.target_depth:
            self.W_multipled = self.W_left @ self.W_right
            # from IPython import embed
            # embed()
            if self.is_left:
                self.W_multipled = self.W_multipled.reshape(-1, 1)[:self.target_dim, :]
            else:
                self.W_multipled = self.W_multipled.reshape(1, -1)[:, :self.target_dim]
        else:
            self.W_left.generate_parameters()
            self.W_right.generate_parameters()
            self.W_multipled = self.W_left.W_multipled @ self.W_right.W_multipled
            if self.is_left:
                self.W_multipled = self.W_multipled.reshape(-1, 1)[:self.target_dim, :]
            else:
                self.W_multipled = self.W_multipled.reshape(1, -1)[:, :self.target_dim]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        self.generate_parameters()
        # if self.input_dim > self.output_dim:
        W = self.W_multipled.reshape(self.input_dim, self.output_dim)

        # else:
            # W = self.W_multipled.reshape(self.output_dim, self.input_dim).transpose(0,1)
        writer.add_scalar("weight/W_multipled", torch.mean(W), self.glb_iter)
        self.glb_iter+=1
        # W = self.W_left*self.W_right
        # from IPython import embed
        # embed()


        output = torch.matmul(input=x, other=W)
        if self.bias:
            output += self.b
        return output

# class HierachicalLowRankLinear(torch.nn.Module):
#     def __init__(self, input_dim: int, output_dim: int, rank: int = 1,
#         bias: bool = True, w_init: str = "glorot-uniform"):
#         super(LowRankLinear, self).__init__()
#         self.input_dim = input_dim
#         self.output_dim = output_dim 
#         self.rank = rank
#         self.bias = bias
#         self.w_init = w_init
#         self.W_left = 
#         self.W_right = nn.Parameter(torch.Tensor(size=(rank, output_dim)), requires_grad=True)
#         if bias:
#             self.b = nn.Parameter(torch.Tensor(output_dim))
#         self.reset_parameters()
    
#     def reset_parameters(self):
#         if self.bias:
#             self.b.data = torch.zeros_like(self.b.data)
#         if self.w_init == "glorot-uniform": 
#             self.W_left.data = glorot_uniform(self.W_left.data) 
#             self.W_right.data = glorot_uniform(self.W_right.data)          
#         elif self.w_init == "glorot-normal":
#             self.W_left.data = glorot_normal(self.W_left.data)
#             self.W_right.data = glorot_normal(self.W_right.data)
#         else:
#             raise ValueError

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         W = self.W_left*self.W_right
#         output = torch.matmul(input=x, other=W)
#         if self.bias:
#             output += self.b
#         return output


class TopoAdapterLayer(nn.Module):
    """This is the low-rank adapter, in which each adapter is composed of two rank-one matrices.
    """
    def __init__(self, 
                 reduction_factor=32, 
                 non_linearity="gelu_new",
                 low_rank_w_init="glorot-uniform", 
                 low_rank_rank=2,
                 linear_type="random_mapping",
                 device=None):
        super().__init__()
        # self.glb_iter=0
        self.reduction_factor = reduction_factor
        self.non_linearity = non_linearity
        self.low_rank_w_init = low_rank_w_init
        self.low_rank_rank = low_rank_rank
        self.device = device
        self.linear_type = linear_type
        self.instantiated = False

    
    def instantiate(self, hidden_dim):

        self.down_sample_size = hidden_dim // self.reduction_factor
        self.activation = Activations(self.non_linearity.lower()).to(self.device)
        if self.linear_type == "random_mapping":
            self.down_sampler = RemappingLinear(hidden_dim, self.down_sample_size,
                                            w_init=self.low_rank_w_init,
                                            rank=self.low_rank_rank).to(self.device)
            self.up_sampler = RemappingLinear(self.down_sample_size, hidden_dim,
                                            w_init=self.low_rank_w_init,
                                            rank=self.low_rank_rank).to(self.device)

        
        elif self.linear_type == "degraded":
            self.down_sampler = DegradedLinear(hidden_dim, self.down_sample_size,
                                            w_init=self.low_rank_w_init,
                                            rank=self.low_rank_rank).to(self.device)
            self.up_sampler = DegradedLinear(self.down_sample_size, hidden_dim,
                                            w_init=self.low_rank_w_init,
                                            rank=self.low_rank_rank).to(self.device)
        
        elif self.linear_type == "hierachical_low_rank":
            self.down_sampler = HierarchicalLowRankSubLayer(input_dim=hidden_dim, 
                                                output_dim=self.down_sample_size,
                                                w_init=self.low_rank_w_init,
                                                rank=self.low_rank_rank,
                                                is_left=True).to(self.device)
            self.up_sampler = HierarchicalLowRankSubLayer(input_dim=self.down_sample_size, 
                                                output_dim=hidden_dim,
                                                w_init=self.low_rank_w_init,
                                                rank=self.low_rank_rank,
                                                is_left=False).to(self.device)

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
        # else:
            # writer.add_scalar("weight/W_multipled", torch.mean(W), self.glb_iter)
            # self.glb_iter+=1
            

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






class TopoAdapterModel(DeltaBase, nn.Module):
    r"""
    """
    def __init__(self, 
                 common_structure = False,
                 structure_mapping = None,
                 reduction_factor = 32,
                 non_linearity = "gelu_new",
                 low_rank_w_init = "glorot-uniform", 
                 linear_type = "random_mapping",
                 low_rank_rank = 1,
                 ):
            
        DeltaBase.__init__(self, common_structure=common_structure, structure_mapping=structure_mapping)
        nn.Module.__init__(self)
        self.reduction_factor = reduction_factor
        self.non_linearity = non_linearity
        self.low_rank_w_init = low_rank_w_init
        self.low_rank_rank = low_rank_rank
        self.linear_type = linear_type
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
        adapterlayer = TopoAdapterLayer(reduction_factor = self.reduction_factor,
                                      non_linearity = self.non_linearity,
                                      low_rank_w_init = self.low_rank_w_init, 
                                      low_rank_rank = self.low_rank_rank,
                                      linear_type = self.linear_type,
                                      device=module_device)
        self.delta_modules.append(adapterlayer)  
        return adapterlayer
    