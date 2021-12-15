

from inspect import signature
from typing import Optional
from opendelta.utils.utils import *
import torch.nn as nn
import torch
from functools import wraps
from decorator import decorate
class DeltaBase(object):
    """Not rigorous currently
    """
    def __init__(self):
        pass
    
    def freeze_plm(self, plm):
        for n, p in plm.named_parameters():
            p.requires_grad = False
    
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

    def find_module(self, root_module: nn.Module, key:str):
        r"""Find the module using a name given by module.named_parameters() methods.
        Return both the parent reference and the child name and reference.
        """
        sub_keys = key.split(".")
        parent_module = root_module
        for sub_key in sub_keys[:-1]:
            parent_module = getattr(parent_module, sub_key)
        module = getattr(parent_module, sub_keys[-1])
        return parent_module, sub_keys[-1], module

    def replace_module(self,
                      parent_module: nn.Module, 
                      children_name: str,
                      replacedmodule: Optional[nn.Module]=None):
        r"""Replace a module using the reference of its parent module.
        """
        raise NotImplementedError
    
    def modify_module(self, module: nn.Module):
        r"""Modify the inside parameteres of a module.
        """
        raise NotImplementedError

    def insert_sequential_module(self, module, pre_caller=None, post_caller=None):
        r"""insert a module (previous not exists in the code base) before a module. Specifically, it modifies the forward 
        function of the original module to  firstly pass the arguments into the new module's forward function and then pass
        it into the original ones. The new module can also be inserted after the original module with similar mechanism. 

        When implementing the new module , researchers should be aware of the components of arguments of the original module's forward function.
        """
        def _caller(_org_func, _pre_caller, _post_caller,  *args, **kwargs):
            if _pre_caller is not None:
                args, kwargs = _pre_caller(*args, **kwargs)
            ret = _org_func(*args, **kwargs)
            if _post_caller is not None:
                ret = _post_caller(ret)
            return ret

        if hasattr(module.forward, "__wrapped__"):
            raise RuntimeWarning("The forward function might have been wrapped by a decorator, is it intended?")
        module.forward = decorate(module.forward, _caller, extras=(pre_caller, post_caller), kwsyntax=True) # decorator.decorate helps preserving the functions metadata (signature, etc.).
    

    def insert_parrellel_module(self, ):
        # TODO
        pass


        

