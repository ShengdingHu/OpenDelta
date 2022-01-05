

import os
from opendelta.utils.signature import signature
from typing import Optional, Union
from opendelta.utils.cuda import get_device
from opendelta.utils.utils import *
import torch.nn as nn
import torch
from functools import wraps
from decorator import decorate
from opendelta.utils.structure_mapping import transform


def is_leaf_module(module):
    r"""Whether the module is a leaf module
    """
    return len([n for n,_ in module.named_children()]) == 0

def non_module_param(module: nn.Module):
    module_names = [n for n, _ in module.named_modules()]
    ret = []
    for n, p in module.named_parameters():
        if not is_child_key(n, module_names):
            ret.append((n,p))
    return ret



class DeltaBase(object):
    """Not rigorous currently
    """
    def __init__(self, 
                 common_structure=False,
                 structure_mapping=None
                 ):
        self.common_structure = common_structure
        self.structure_mapping = structure_mapping
        if self.common_structure and self.structure_mapping is None:
            raise RuntimeError("Using common structure but the structure mapping is None")

    def __call__(self, 
                 module: nn.Module, 
                 modified_keys: List[str],
                 is_regex: Optional[bool]=False,
                 registration_name: Optional[str] = "deltas"
                ) -> nn.Module:
        r"""modify the modules into delta models.

        Args:

        Returns:

        """
        self.plm_total_params = sum(p.numel() for p in module.parameters())
        for key, _ in module.named_modules():
            if self.find_key(key, modified_keys, is_regex):
                # print("find key",key)
                self.update_module(module, key)
        # if the delta parameters are not contained in the original models parameters
        # we need to register it to the module
        self.register_delta_if_new(module, registration_name)
        # mark the paratmers that are the delta parameters for easily 
        # extracting the delta_paramters.
        # This is important if the delta parameters are contained in the
        # original models parameters
        self.mark_as_delta()
        return module
    
    def register_delta_if_new(self, module: nn.Module, registration_name: Optional[str] = "deltas"):
        setattr(module, registration_name, self)

    
    def mark_as_delta(self, module: nn.Module=None,):
        if module is None:
            module=self
        for p in module.parameters():
            setattr(p, "_is_delta", True)
    
    def update_module(self, module: nn.Module, key: str):
        r"""Different in each delta models. 
        """
        raise NotImplementedError
    
    
    def freeze_module(self, module: nn.Module, exclude=["deltas"], is_regex=False, prefix=""):
        r"""Freeze the parameters of plm. Leave the parameters in exclude untouched.
        deltas module should be filtered with `_is_delta` attributes because it may have parameter sharing to the main model, (e.g., bias term)
        """
        if is_leaf_module(module):
            for n, p in module.named_parameters():
                if self.find_key(".".join([prefix,n]), exclude, is_regex=is_regex, only_tail=True):
                    continue
                if "deltas" not in exclude or (not (hasattr(p, "_is_delta") and getattr(p, "_is_delta"))):
                    p.requires_grad = False
            return 
        else:
            for n, c in module.named_children():
                if self.find_key(".".join([prefix,n]), exclude, is_regex=is_regex, only_tail=True): # if found, untouch the parameters
                    continue
                else: # firstly freeze the non module params, then go deeper.
                    params = non_module_param(module)
                    for n, p in params:
                        if "deltas" not in exclude or (not (hasattr(p, "_is_delta") and getattr(p, "_is_delta"))):
                            p.requires_grad = False
                    self.freeze_module(c, prefix=".".join([prefix,n]), exclude=exclude, is_regex=is_regex)



    def find_key(self, key: str, target_list: List[str], is_regex: bool, only_tail=True):
        r""" Check whether any target string is in the key or in the tail of the key. 

        Args:
            key (:obj:`str`) The key which might be a referencing name to a submodule in the module, e.g. bert.embeddings.word_embeddings
            target (:obj:`List[str]`) The list of target string e.g. ["attention", "word_embeddings", "ffn.out"] 
            is_regex (:obj:`bool`) whether the syntax in the target_list is plain text or regular expression.

        Returns: 
            bool
        """
        if self.common_structure:
            key = transform(key, self.structure_mapping, strict=False)
        if key is None:
            return False
        if is_regex:
            try:
                re.compile(key)
            except re.error:
                print(f"Non valid regular expression. {key}")
            if only_tail:
                # print("here")
                # # from IPython import embed
                # # embed()
                return endswith_in_regex(key, target_list)
            else:
                return substring_in_regex(key, target_list)
        else:
            if only_tail:
                return endswith_in(key, target_list)
            else:
                return substring_in(key, target_list)

    def pseudo_data_to_instantiate(self, module):
        r"""Create a pseudo_data into the module to know the dimemsion of each tensor in the computation graph.
        #TODO: To test more data input format, i.e. may need to pass more than inputs/decoder_input_ids.
        """
        device = get_device(module)
        pseudo_input = torch.tensor([[0,0]]).to(device)
        if "decoder_input_ids" in  signature(module.forward).args:
            module(pseudo_input, decoder_input_ids = pseudo_input)
        else:
            module(pseudo_input)

    def trainable_parameters_names(self, module: Optional[nn.Module]=None):
        if module is None:
            module = self
        return [n for n,p in module.named_parameters() if p.requires_grad]
    
    def frozen_parameters_names(self, module: Optional[nn.Module]=None):
        if module is None:
            module = self
        return [n for n,p in module.named_parameters() if not p.requires_grad]

    def trainable_parameters(self,module: Optional[nn.Module]=None):
        if module is None:
            module = self
        return [p for n,p in module.named_parameters() if p.requires_grad]


    def num_trainable_parameters(self, module: Optional[nn.Module]=None):
        if module is None:
            module = self
        pnum_tot = 0
        for param in module.parameters():
            if param.requires_grad:
                pnum_tot += param.numel()
        return pnum_tot
    
    def num_frozen_parameters(self, module: Optional[nn.Module]=None):
        if module is None:
            module = self
        pnum_tot = 0
        for param in module.parameters():
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

        Returns:
            nn.Module: A reference to the parent module of the replaced module
            str: The key of the replaced module relevant to its parent module
            nn.Module: The replaced module. 
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

    def set_active_state_dict(self, module: nn.Module):
        r"""modify the state_dict of the model

        filter_func: only the parameters that requires grad is keeped in the state_dict
        """
        def _caller(_org_func, excludes,  *args, **kwargs):
            state_dict = _org_func(*args, **kwargs)
            keys = list(state_dict.keys())
            for n  in keys:
                if n in excludes:
                    state_dict.pop(n)
            return state_dict
        excludes = self.frozen_parameters_names(module)
        
        if hasattr(module.state_dict, "__wrapped__"):
            raise RuntimeWarning("The forward function might have been wrapped by a decorator, is it intended?")
        module.state_dict = decorate(module.state_dict, _caller, extras=(excludes,), kwsyntax=True) # decorator.decorate helps preserving the functions metadata (signature, etc.).
    

    def from_pretrained(self, module:nn.Module, pretrained_deltas_name_or_path: Optional[Union[str, os.PathLike]], *model_args, **kwargs):
        r"""Todo: currenlty, simply load the state_dict and instantiate the original model 
        """
        state_dict = torch.load(os.path.join(pretrained_deltas_name_or_path))
        module.load_state_dict(state_dict, strict=False)
        
    


        

