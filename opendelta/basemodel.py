

from typing import Optional
from opendelta.utils.utils import *
import torch.nn as nn
import torch

class DeltaBase(object):
    """Not rigorous currently
    """
    def __init__(self):
        pass
    
    def new_module_like(self, module: nn.Module):
        raise NotImplementedError
    
    @property
    def trainable_parameters_names(self,):
        return [n for n,p in self.named_parameters() if p.requires_grad]

    @property
    def trainable_parameters(self,):
        return [p for n,p in self.named_parameters() if p.requires_grad]

    @property
    def num_trainable_parameters(self,):
        pnum_tot = 0
        for param in self.parameters():
            if param.requires_grad:
                pnum_tot += param.numel()
        return pnum_tot

    @property
    def num_additional_frozen_parameters(self,):
        pnum_tot = 0
        for param in self.parameters():
            if not param.requires_grad:
                pnum_tot += param.numel()
        return pnum_tot
    
    def load_state_dict(self, path):
        pass

    def from_pretrained(self, path, config):
        pass

    def modify_module(self,
                      module: nn.Module, 
                      key: str,
                      replacedmodule: Optional[nn.Module]=None):
        r"""Replace a module's child module using the module's reference name.
        If the replacemodule is None, it will call self.new_module_like method which are
        different for different objects. 
        """
        sub_keys = key.split(".")
        ptr = module
        for sub_key in sub_keys[:-1]:
            ptr = getattr(ptr, sub_key)
        if replacedmodule is None:
            replacedmodule = self.new_module_like(getattr(ptr, sub_keys[-1]))
        setattr(ptr, sub_keys[-1], replacedmodule)

    @staticmethod
    def modify_module_input_output(module, modify_name, pre_func=lambda x:x, post_func=lambda x:x):
        r"""Not tested yet.
        """
        def decorator_func(_org_func=None, _pre_func=None, _post_func=None):
            def func_wrapper(*args,**kwargs):
                new_args, new_kwargs = _pre_func(*args, **kwargs)
                ret = _org_func(*new_args, **new_kwargs)
                new_ret = _post_func(ret)
                return new_ret
            return func_wrapper
        for name, m in module.named_modules(): 
            if name == modify_name:
                module.forward = decorator_func(_org_func=m.forward, _pre_func=pre_func, _post_func=post_func)


