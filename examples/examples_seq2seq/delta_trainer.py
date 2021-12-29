from packaging import version
import torch
from torch import nn
from typing import Any, Dict, List, Optional, Tuple, Union

from torch.utils.data.dataset import Dataset
from transformers import Seq2SeqTrainer
from examples_seq2seq.trainers.trainer import BaseTrainer 

    # if is_sagemaker_mp_enabled():
#     import smdistributed.modelparallel.torch as smp

# from transformers.trainer_utils import ShardedDDPOption

# if is_fairscale_available():
#     dep_version_check("fairscale")
#     import fairscale
#     from fairscale.nn.data_parallel import FullyShardedDataParallel as FullyShardedDDP
#     from fairscale.nn.data_parallel import ShardedDataParallel as ShardedDDP
#     from fairscale.nn.wrap import auto_wrap
#     from fairscale.optim import OSS
#     from fairscale.optim.grad_scaler import ShardedGradScaler

from transformers.optimization import Adafactor, AdamW, get_scheduler
from transformers.trainer_pt_utils import get_parameter_names, is_sagemaker_mp_enabled
from transformers.integrations import is_fairscale_available



if version.parse(torch.__version__) >= version.parse("1.6"):
    from torch.cuda.amp import autocast


class DeltaTrainer(Seq2SeqTrainer, BaseTrainer):
    def __init__(self, train_dataset_sizes=None, delta_args=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_dataset_sizes = train_dataset_sizes
        self.delta_args = delta_args

    def evaluate(
        self,
        eval_dataset: Optional[Dict[str, Dataset]] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        max_length: Optional[int] = None,
        num_beams: Optional[int] = None,
    ) -> Dict[str, float]:
        # TODO: this also needs to be set per dataset
        self._max_length = max_length
        self._num_beams = num_beams
        return super().evaluate(eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)


    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on :obj:`model` using obj:`inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`):
                The model to evaluate.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (:obj:`bool`):
                Whether or not to return the loss only.

        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
            labels (each being optional).
        """
        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)
        gen_kwargs = {
            "max_length": self._max_length if self._max_length is not None else self.model.config.max_length,
            "num_beams": self._num_beams if self._num_beams is not None else self.model.config.num_beams,
        }
        generated_tokens = self.model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            **gen_kwargs,
        )
        # in case the batch is shorter than max length, the output should be padded
        if generated_tokens.shape[-1] < gen_kwargs["max_length"]:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_length"])

        with torch.no_grad():
            if self.use_amp:
                with autocast():
                    outputs = model(**inputs)
            else:
                outputs = model(**inputs)
            if has_labels:
                if self.label_smoother is not None:
                    loss = self.label_smoother(outputs, inputs["labels"]).mean().detach()
                else:
                    loss = (outputs["loss"] if isinstance(outputs, dict) else outputs[0]).mean().detach()
            else:
                loss = None

        if self.args.prediction_loss_only:
            return (loss, None, None)

        labels = inputs["labels"]
        if labels.shape[-1] < gen_kwargs["max_length"]:
            labels = self._pad_tensors_to_max_len(labels, gen_kwargs["max_length"])

        return (loss, generated_tokens, labels)
    
    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through :obj:`optimizers`, or subclass and override this method in a subclass.
        """
        if self.optimizer is None:
            decay_parameters = get_parameter_names(self.model, [nn.LayerNorm])
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if n in decay_parameters and p.requires_grad],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if n not in decay_parameters and p.requires_grad],
                    "weight_decay": 0.0,
                },
            ]
            # from IPython import embed
            # print("In line 145")
            # embed()
            # print("optimized", optimizer_grouped_parameters)
            optimizer_cls = Adafactor if self.args.adafactor else AdamW
            if self.args.adafactor:
                optimizer_cls = Adafactor
                optimizer_kwargs = {"scale_parameter": False, "relative_step": False}
            else:
                optimizer_cls = AdamW
                optimizer_kwargs = {
                    "betas": (self.args.adam_beta1, self.args.adam_beta2),
                    "eps": self.args.adam_epsilon,
                }
            optimizer_kwargs["lr"] = self.delta_args.delta_lr
            # if self.sharded_ddp == ShardedDDPOption.SIMPLE:
            #     self.optimizer = OSS(
            #         params=optimizer_grouped_parameters,
            #         optim=optimizer_cls,
            #         **optimizer_kwargs,
            #     )
            # else:
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        # if is_sagemaker_mp_enabled():
        #     self.optimizer = smp.DistributedOptimizer(self.optimizer)
        return self.optimizer

    
    
    
    
