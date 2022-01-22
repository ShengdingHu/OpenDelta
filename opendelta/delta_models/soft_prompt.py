from sys import prefix

from opendelta.utils.signature import get_arg_names, get_arg_names_inside_func
from opendelta.utils.utils import *
from opendelta.utils.cuda import get_device
from opendelta.basemodel import DeltaBase
from typing import *
import torch
import torch.nn as nn
from opendelta import BaseDeltaConfig
from decorator import decorate
import torch.nn.functional as F

class SoftPromptConfig(BaseDeltaConfig):
    r"""
    This is the configuration class to store the configuration of a [`LoraModel`]

    """
    def __init__(
        self, 
        soft_token_num=100,
        **kwargs
    ): 
        super().__init__(**kwargs)
        arg_names = get_arg_names_inside_func(self.__init__)
        for arg_name in arg_names:
            if not hasattr(self, arg_name): # the arg has not been registered in parent config
                setattr(self, arg_name, locals()[arg_name])


class SoftTemplatePre(nn.Module):
    r"""This is the implementation of `The Power of Scale for Parameter-Efficient
    Prompt Tuning <https://arxiv.org/pdf/2104.08691v1.pdf>`_ . Similar to :obj:`PrefixTuningTemplate`,
    This template also does not need any textual template. Addition tokens are directly
    concatenated into the input ids. There are two initializations of the new tokens. 
    (1). random initialization. (2) initialize with the tokens of the plm (We simply take 
    the first n_tokens similar to their implementation).

    Note that this template can be simply achieved by :obj:`SoftManualTemplate`, in which
    you set `n_token` <soft> tokens template before the <text_a> will give the same result.
    """

    def __init__(self,
                 model,
                 raw_embedding,
                 mask_id,
                 is_encoder_decoder,
                 soft_token_num: int = 100,
                 random_range: Optional[float] = 0.5,
                ):
        super().__init__()
        self.raw_embedding = raw_embedding
        self.random_range = random_range
        self.num_tokens = soft_token_num
        self.mask_id = mask_id
        self.is_encoder_decoder = is_encoder_decoder

        if self.num_tokens>0:
            self.generate_parameters()


    def generate_parameters(self) -> None:
        """
        generate parameters needed for soft tokens embedding in soft-prompt
        for soft tokens, use a new embedding layer which is initialized with their corresponding embedding of hard tokens
        """
        soft_embeds = torch.FloatTensor(self.num_tokens, self.raw_embedding.weight.size(1)).uniform_(-self.random_range, self.random_range)
        self.soft_embeds = nn.Parameter(soft_embeds, requires_grad=True)

    
    def forward(self, *args, **batch):
        """
        Convert input_ids to inputs_embeds
        for normal tokens, use the embedding layer of PLM
        for soft tokens, use a new embedding layer which is initialized with their corresponding embedding of hard tokens
        """
        inputs_embeds = self.raw_embedding(batch['input_ids'])
        batch_size = inputs_embeds.size(0)
        if self.num_tokens>0:
            soft_embeds = self.soft_embeds.repeat(batch_size, 1, 1)
            inputs_embeds = torch.cat([soft_embeds, inputs_embeds], 1)

        if self.is_encoder_decoder:
            loss_selector = torch.where(batch["decoder_input_ids"] == self.mask_id)
        else:
            loss_selector = torch.where(batch["input_ids"] == self.mask_id)
        batch.pop('input_ids')
        batch['inputs_embeds'] = inputs_embeds
        if 'attention_mask' in batch and self.num_tokens>0:
            am = batch['attention_mask']
            batch['attention_mask'] = torch.cat([torch.ones((batch_size,self.num_tokens), dtype = am.dtype,device=am.device), am], dim=-1)
        return loss_selector, batch
    
class SoftTemplatePost(nn.Module):
    def __init__(self,
                 verbalizer,
                 soft_token_num: int = 100,
                 shift: bool = True,
                ):
        super().__init__()
        self.soft_token_num = soft_token_num
        self.shift = shift
        self.verbalizer = torch.LongTensor(verbalizer).cuda()

    def forward(self, outputs, loss_selector, targets, **batch):
        r"""Post processing the outputs of language models according
        to the need of template. Most templates don't need post processing,
        The template like SoftTemplate, which appends soft template as a module
        (rather than a sequence of input tokens) to the input,
        should remove the outputs on these positions to keep the seq_len the same
        """
        logits = outputs.logits
        if self.shift:
            logits = logits[:, self.soft_token_num:, :]
        logits = logits[loss_selector]
        logits = logits.index_select(dim=-1, index=self.verbalizer)
        loss = F.cross_entropy(logits, targets)
        return {"loss": loss, "logits": logits}


class SoftPromptModel(DeltaBase):
    r""" Support insert prefix token before each layer. For example, layer 3 4 6 10 and other layer untouched. 

    Args:
        emb_dim (:obj:`int`) The embedding dim of the reparameterization model.

    """
    config_class = SoftPromptConfig
    delta_type = "soft_prompt"

    def __init__(self, 
                 backbone_model: nn.Module,
                 verbalizer,
                 mask_id,
                 soft_token_num=100,
                 modified_modules: Optional[bool] = None,
                 common_structure: Optional[bool] = None,
                 registration_name: Optional[str] = "deltas",
                ):
        DeltaBase.__init__(self, 
                           backbone_model = backbone_model,
                           modified_modules=["root"],
                           common_structure = common_structure,
                           registration_name = registration_name,
                          )

        arg_names = get_arg_names_inside_func(self.__init__)
        for arg_name in arg_names:
            if not hasattr(self, arg_name): # not registered in parent class
                setattr(self, arg_name, locals()[arg_name])

        self.softlayer_pre = SoftTemplatePre(
            backbone_model,
            soft_token_num = self.soft_token_num,
            raw_embedding = backbone_model.get_input_embeddings(),
            is_encoder_decoder = backbone_model.config.is_encoder_decoder,
            mask_id = mask_id
        )
        self.softlayer_post = SoftTemplatePost(
            soft_token_num = self.soft_token_num,
            shift = not backbone_model.config.is_encoder_decoder,
            verbalizer = verbalizer
        )
        self.insert_sequential_module(backbone_model, pre_caller=self.softlayer_pre.forward, post_caller=self.softlayer_post)
        self.mark_as_delta()
        self.register_delta_if_new(backbone_model, registration_name)

    def insert_sequential_module(self, module, pre_caller=None, post_caller=None):
        r"""insert a module (previous not exists in the code base) before a module. Specifically, it modifies the forward 
        function of the original module to  firstly pass the arguments into the new module's forward function and then pass
        it into the original ones. The new module can also be inserted after the original module with similar mechanism. 

        When implementing the new module , researchers should be aware of the components of arguments of the original module's forward function.
        """
        def _caller(_org_func, _pre_caller, _post_caller,  *args, **batch):
            if _pre_caller is not None:
                loss_selector, batch = _pre_caller(*args, **batch)
            labels = batch["labels"]
            batch["labels"] = None
            ret = _org_func(**batch)
            if _post_caller is not None:
                ret = _post_caller(ret, loss_selector, labels, **batch)
            return ret

        if hasattr(module.forward, "__wrapped__"):
            raise RuntimeWarning("The forward function might have been wrapped by a decorator, is it intended?")
        module.forward = decorate(module.forward, _caller, extras=(pre_caller, post_caller), kwsyntax=True) # decorator.decorate helps preserving the functions metadata (signature, etc.).
