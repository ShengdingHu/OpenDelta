from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class DeltaArguments:
    """Defines the delta models parameters."""

    delta_type: Optional[str] = field(default="none", 
                                      metadata={"help": "The type of delta modules to use"})
    common_structure: Optional[bool] = field(default=False, metadata={"help": "Whether to use the common structure mapping."
                                                                    "Not that this common structure only support several commonly used models"
                                                                    "Currently include t5 gpt2 distilbert roberta."
                                                                    "Use other model with common structure, the user should explicitly know write"
                                                                    "the structure mapping first. However, this feature is useful when measureing/studying the"
                                                                    "generalization ability of a delta model across different pretrained models. "})
    modified_modules: Optional[List[str]] = field(default=None,
                                      metadata={"help": "The list of names of the modules that need to insert deltas."})
    unfreeze_modules: Optional[List[str]] = field(default_factory=lambda : ["deltas"],
                                      metadata={"help": "The list of names of the unfreeze modules"})

    pretrained_delta_path: Optional[str] = field(default="",
                                     metadata = {"help": "The path of the saved delta modules checkpoints"})

    lora_alpha: Optional[int] = field(default=8, 
                                      metadata={"help": "The scaling parameter in lora model"})
    lora_rank : Optional[int] = field(default=8,
                                      metadata={"help": "Rank of the lora model"})
    lora_dropout: Optional[float] = field(default=0.0,
                                      metadata={"help": "The dropout rate in loar.linear"})
                
    