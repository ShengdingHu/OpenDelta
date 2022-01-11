

import os
from opendelta.delta_configs import BaseDeltaConfig
from opendelta.utils.model_md5 import gen_model_hash
from opendelta.utils.signature import get_arg_names, signature
from typing import Optional, Union
from opendelta.utils.cuda import get_device
from opendelta.utils.utils import *
import torch.nn as nn
import torch
from functools import wraps
from decorator import decorate
from opendelta.utils.structure_mapping import transform
from transformers.file_utils import PushToHubMixin
from transformers.deepspeed import deepspeed_config, is_deepspeed_zero3_enabled
from opendelta import SaveLoadMixin
from opendelta import logging

logger = logging.get_logger(__name__)

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


def load_structure_mapping_according_to_backbone(backbone_type):
    # backbone_type = 
    return None



class DeltaBase(nn.Module, SaveLoadMixin):
    """The Base class for all delta models. 
    """
    delta_type = ""
    config_class = BaseDeltaConfig
    _keys_to_ignore_on_save = None
    def __init__(self, 
                 backbone_model: nn.Module,
                 modified_modules: Optional[List[str]] = None,
                 registration_name: Optional[str] = "deltas",
                 common_structure=False,
                 ):
        nn.Module.__init__(self)
        self.__dict__["backbone_model"] = backbone_model
        self.backbone_hash = gen_model_hash(backbone_model)
        if modified_modules  is None:
            self.modified_modules = self.default_modified_modules
            self.common_structure = True
        else:
            self.modified_modules = modified_modules
            self.common_structure = common_structure
        if self.common_structure:
            self.structure_mapping = load_structure_mapping_according_to_backbone(type(self.backbone_model))
        else:
            self.structure_mapping = None
        self.registration_name = registration_name
        if self.common_structure and self.structure_mapping is None:
            raise RuntimeError("Using common structure but the structure mapping is None")
        
    def forward(self, *args, **kwargs) -> "RuntimeError":
        r""" 
            .. warning::

                Removed method. As the model is a delta model, which should be attached to a backbone model \
                and can't forward any data by itself. Please using the backbone model's forward function \
                after attach the delta model to the backbone.
        """
        raise RuntimeError("This is a delta model, which should be attached to a backbone model \
            and can't forward any data by itself. Please using the backbone model's forward function \
            after attach the delta model to the backbone. ")

    @classmethod
    def from_config(cls, config: Union[BaseDeltaConfig, dict], backbone_model: nn.Module, check_hash=True, **kwargs):
        r"""Initialize a delta model from a config object or a dict containing the configs. To temperarily change
        a value in the config, pass it through kwargs. If the config has a backbone model's hash, which means it is
        a finetuned delta model's config, then we will compare the hash in the config and the newly caculated to ensure
        the finedtuned delta model is trained on the passed backbone_model. Pass `check_hash=False` to disable the
        checking.

        Args:
            config: (:obj:`BaseDeltaConfig` or `dict`) A config object or a dict that contains the necessary value to 
                            initialize the delta model.
            backbone_model (:obj:`nn.Module`) A pytorch module that will be pass into the delta model as the backbone 
                    model. modifications will be made in place in the backbone model.
            check_hash (:obj:`bool`, default to `True`) Whether to check hash of the backbone model and the config's 
                            backbone hash. 
            kwargs: Any configurations that are passed to update the config object. 
        """
        backbone_hash = gen_model_hash(backbone_model)
        if check_hash and hasattr(config, "backbone_hash") and \
                          config.backbone_hash is not None and \
                          config.backbone_hash != backbone_hash:
            raise RuntimeError("The config has an hash of the backbone model, and is \
                            different from the hash of the loaded model. This indicates a mismatch \
                            between the backbone model that the delta checkpoint is based on and \
                            the one you loaded. ")
        supported_keys = get_arg_names(cls.__init__) + get_arg_names(DeltaBase.__init__)
        config_dict = config.to_dict()
        for key in list(config_dict.keys()):
            if key not in supported_keys:
                config_dict.pop(key)
        return cls(backbone_model, **config_dict)


    def add_all_delta_to_backbone(self, 
                 backbone: nn.Module, 
                 modified_modules: List[str],
                 registration_name: Optional[str] = "deltas",
                ) -> nn.Module:
        r"""modify the modules into delta models.

        Args:

        Returns:

        """
        self.plm_total_params = sum(p.numel() for p in backbone.parameters())
        for key, _ in backbone.named_modules():
            if self.find_key(key, modified_modules):
                # print("find key",key)
                self.update_module(backbone, key)
        # if the delta parameters are not contained in the original models parameters
        # we need to register it to the module
        self.register_delta_if_new(backbone, registration_name)
        # mark the paratmers that are the delta parameters for easily 
        # extracting the delta_paramters.
        # This is important if the delta parameters are contained in the
        # original models parameters
        self.mark_as_delta()
        return backbone
    
    def register_delta_if_new(self, module: nn.Module, registration_name: Optional[str] = "deltas"):
        r"""
        """
        setattr(module, registration_name, self)

    
    def mark_as_delta(self, module: nn.Module=None,):
        r""" Mark a model's all parameters as delta parameters by setting a "_is_delta"  attribute to each of them.
        Generally, it is used after creating the delta modules.
        Args:
            module (:obj:`nn.Module`) The module to mark as delta.
        """
        if module is None:
            module=self
        for p in module.parameters():
            setattr(p, "_is_delta", True)
    
    def update_module(self, module: nn.Module, key: str):
        r"""Update a module specified by :obj:`key`. The method is reimplemented 
        """
        raise NotImplementedError
    
    
    def freeze_module(self, module: Optional[nn.Module] = None, exclude=["deltas"], set_state_dict: Optional[bool]=True, prefix=""):
        r"""Freeze the parameters of plm. Leave the parameters in exclude untouched.
        deltas module should be filtered with `_is_delta` attributes because it may have parameter sharing to the main model, (e.g., bias term)
        """
        if module is None:
            module = self.backbone_model
        if is_leaf_module(module):
            for n, p in module.named_parameters():
                if self.find_key(".".join([prefix,n]), exclude, only_tail=True):
                    continue
                if "deltas" not in exclude or (not (hasattr(p, "_is_delta") and getattr(p, "_is_delta"))):
                    p.requires_grad = False
            return 
        else:
            for n, c in module.named_children():
                if self.find_key(".".join([prefix,n]), exclude, only_tail=True): # if found, untouch the parameters
                    continue
                else: # firstly freeze the non module params, then go deeper.
                    params = non_module_param(module)
                    for n, p in params:
                        if "deltas" not in exclude or (not (hasattr(p, "_is_delta") and getattr(p, "_is_delta"))):
                            p.requires_grad = False
                    self.freeze_module(c, prefix=".".join([prefix,n]), exclude=exclude)
        
        if set_state_dict:
            self.set_active_state_dict(module)



    def find_key(self, key: Union[str, re.Pattern], target_list: List[str], only_tail=True):
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
        if isinstance(key, re.Pattern): # TODO: unit test needed ERROR
            if only_tail:
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
    

    # def from_finetuned(self, module:nn.Module, pretrained_deltas_name_or_path: Optional[Union[str, os.PathLike]], *model_args, **kwargs):
    #     r"""Todo: currenlty, simply load the state_dict and instantiate the original model 
    #     """
    #     state_dict = torch.load(os.path.join(pretrained_deltas_name_or_path))
    #     module.load_state_dict(state_dict, strict=False)

    @classmethod
    def _from_config(cls, config, **kwargs):
        r"""
        All context managers that the model should be initialized under go here.
        Args:
            torch_dtype (`torch.dtype`, *optional*):
                Override the default `torch.dtype` and load the model under this dtype.
        """
        torch_dtype = kwargs.pop("torch_dtype", None)

        # override default dtype if needed
        dtype_orig = None
        if torch_dtype is not None:
            dtype_orig = cls._set_default_torch_dtype(torch_dtype)

        if is_deepspeed_zero3_enabled(): # TODO: to check compatibility with deepspeed
            import deepspeed

            logger.info("Detected DeepSpeed ZeRO-3: activating zero.init() for this model")
            # this immediately partitions the model across all gpus, to avoid the overhead in time
            # and memory copying it on CPU or each GPU first
            with deepspeed.zero.Init(config_dict_or_path=deepspeed_config()):
                model = cls(config, **kwargs)
        else:
            model = cls(config, **kwargs)

        # restore default dtype if it was modified
        if dtype_orig is not None:
            torch.set_default_dtype(dtype_orig)

        return model

    def create_config_from_model(self, ):
        # common_attributes
        config = self.config_class()
        config_keys = signature(config.__init__)[0] + signature(super(self.config_class, config).__init__)[0]

        for key in config_keys:
            val = getattr(self, key) if hasattr(self, key) else None
            setattr(config, key, val)
        config.delta_type = self.delta_type
        self.config = config
    
    def load_state_dict_into_backbone(self, backbone_model: nn.Module = None, state_dict: dict = {}):
        if backbone_model is None:
            backbone_model = self.backbone_model
        self.backbone_model.load_state_dict(state_dict, strict=False)




        
    


        

