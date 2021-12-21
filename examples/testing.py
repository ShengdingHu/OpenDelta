import transformers
import torch
from datasets import load_dataset, load_metric
from datasets import load_from_disk
from datasets import ClassLabel, Sequence
import random
import pandas as pd
from IPython.display import display, HTML
from transformers import AutoTokenizer
import transformers
import argparse

from transformers.utils.dummy_pt_objects import AutoModelForQuestionAnswering


squad_v2 = False
batch_size = 16

parser = argparse.ArgumentParser("")
parser.add_argument("--model_name", default="distilbert")
parser.add_argument("--model_name_or_path", default=f"distilbert-base-uncased")
parser.add_argument("--max_steps", default=20000, type=int)
parser.add_argument("--delta_lr", type=float, default=2e-5)
parser.add_argument("--delta_model", type=str, default="lora")
parser.add_argument("--common_structure", action="store_true")
args = parser.parse_args()
datasets = load_from_disk("/home/hushengding/huggingface_datasets/saved_to_disk/squad/")


# This is where the opendelta differs from the original transformers code.
if 't5' in args.model_name_or_path:
    from transformers import T5ForConditionalGeneration
    model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path)
elif "gpt2" in args.model_name_or_path:
    from transformers import GPT2LMHeadModel
    model = GPT2LMHeadModel.from_pretrained(args.model_name_or_path)
elif "distilbert" in args.model_name_or_path:
    from transformers.models.distilbert.modeling_distilbert import DistilBertForQuestionAnswering
    model = DistilBertForQuestionAnswering.from_pretrained(args.model_name_or_path)
elif "bart" in args.model_name_or_path:
    from transformers.models.bart.modeling_bart import BartForQuestionAnswering
    model = BartForQuestionAnswering.from_pretrained(args.model_name_or_path)


from opendelta.utils.visualization import Visualization
from opendelta.utils.structure_mapping import Mappings
vis = Visualization(model)
vis.structure_graph()
if args.delta_model == "lora":
    from opendelta.delta_models.lora import LoraModel
    if not args.common_structure:
        if 'distilbert' == args.model_name:
            delta_model = LoraModel()
            delta_model(model, modified_keys=["k_lin", "3.attention.v_lin","4.attention.v_lin"])
            delta_model.freeze_module(model, exclude=["qa_outputs", "deltas", 'bias'])
        else:
            raise NotImplementedError
    else:
        delta_model = LoraModel(common_structure=True, structure_mapping=Mappings[args.model_name])
        delta_model(model, modified_keys=["attn.k", "attn.v"])
        delta_model.freeze_module(model, exclude=["qa_outputs", "deltas"])

elif args.delta_model == "bias":
    from opendelta.delta_models.bias import BiasModel
    if not args.common_structure:
        if 'distilbert' == args.model_name:
            delta_model = BiasModel()
            delta_model(model, modified_keys=["attention", "ffn"])
            delta_model.freeze_module(model, exclude=["qa_outputs", "deltas", "0.sa_layer_norm", "bias"])
        else:
            raise NotImplementedError
    else:
        delta_model = LoraModel(common_structure=True, structure_mapping=Mappings[args.model_name])
        delta_model(model, modified_keys=["attn", "ff"])
        delta_model.freeze_module(model, exclude=["qa_outputs", "deltas"])


elif args.delta_model == "prefix":
    from opendelta.delta_models.prefix import PrefixModel
    if not args.common_structure:
        if 'distilbert' ==  args.model_name:
            delta_model = PrefixModel()
            delta_model(model, modified_keys=["attention"])
            delta_model.freeze_module(model, exclude=["qa_outputs", "deltas", "q_lin"])
        elif 't5' == args.model_name:
            delta_model = PrefixModel()
            delta_model(model, modified_keys=["SelfAttention"])
            delta_model.freeze_module(model, exclude=["qa_outputs", "deltas"])
        elif 'gpt2' == args.model_name:
            delta_model = PrefixModel()
            delta_model(model, modified_keys=["attn"])
            delta_model.freeze_module(model, exclude=["qa_outputs", "deltas"])
        elif 'bart' == args.model_name:
            delta_model = PrefixModel()
            delta_model(model, modified_keys=["self_attn"])
            delta_model.freeze_module(model, exclude=["qa_outputs", "deltas"])
        else:
            raise NotImplementedError
    else:
        print("use common_structure")
        delta_model = PrefixModel(common_structure=True, structure_mapping=Mappings[args.model_name], reparameterize=False)
        delta_model(model, modified_keys=[r"[0-7]\.attn"], is_regex=True)
        delta_model.freeze_module(model, exclude=["qa_outputs", "deltas"])

    

elif args.delta_model == "adapter":
    from opendelta.delta_models.adapter1 import AdapterModel
    if not args.common_structure:
        if 'distilbert' ==  args.model_name:
            delta_model = AdapterModel()
            delta_model(model, modified_keys=[r"attention", "ffn"], registration_name="deltas")
            delta_model.freeze_module(model, exclude=["qa_outputs", "deltas"])
        elif 'bart' == args.model_name:
            delta_model = AdapterModel()
            delta_model(model, modified_keys=[r"self_attn", "fc2"], registration_name="deltas")
            delta_model.freeze_module(model, exclude=["qa_outputs", "deltas"])
        elif 'gpt2' == args.model_name:
            delta_model = AdapterModel()
            delta_model(model, modified_keys=["attn", "mlp"], registration_name="deltas")
            delta_model.freeze_module(model, exclude=["qa_outputs", "deltas"])
        elif 't5' == args.model_name:
            delta_model = AdapterModel()
            delta_model(model, modified_keys=["SelfAttention", "DenseReluDense"], registration_name="deltas")
            delta_model.freeze_module(model, exclude=["qa_outputs", "deltas"]) 
        else:
            raise NotImplementedError
    else:
        delta_model = AdapterModel(common_structure=True, structure_mapping=Mappings[args.model_name])
        delta_model(model, modified_keys=[r"attn", "ff"], registration_name="deltas" )
        delta_model.freeze_module(model, exclude=["qa_outputs", "deltas"])
        

vis2 = Visualization(model)
vis2.structure_graph()
print(f"Tunable params rate {delta_model.num_trainable_parameters()/delta_model.num_frozen_parameters(model)}")




# optimizer_grouped_parameters = []
# optimizer_grouped_parameters.extend([
#     {'params': delta_model.trainable_parameters, 'weight_decay': 0.01, 'lr': args.delta_lr},
#     {'params': [x for x in model.qa_outputs.parameters() if x.requires_grad], 'weight_decay': 0.01, 'lr': args.delta_lr},
# ])
# or simply
optimizer_grouped_parameters = [
    {'params': [x for x in model.parameters() if x.requires_grad], 'weight_decay': 0.01, 'lr': args.delta_lr},
]

print(f"{delta_model.trainable_parameters_names()}")
print("optimized", [len(x['params']) for x in optimizer_grouped_parameters])

# from transformers.optimization import AdamW, get_linear_schedule_with_warmup
# optimizer = AdamW(optimizer_grouped_parameters) # usually lr = 0.5
# scheduler = get_linear_schedule_with_warmup(
#                 optimizer, 
#                 num_warmup_steps=100, num_training_steps=100000) # usually num_warmup_steps is 500









