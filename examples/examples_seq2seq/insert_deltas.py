from examples_seq2seq.delta_args import DeltaArguments
from examples_seq2seq.trainers.model_args import ModelArguments
from opendelta.utils.visualization import Visualization
from opendelta.utils.structure_mapping import Mappings
import torch.nn as nn



def insert_deltas(model, model_args: ModelArguments, delta_args: DeltaArguments):
    ckpt_name = model_args.model_name_or_path.split("/")[-1]
    print(ckpt_name)
    vis = Visualization(model)
    vis.structure_graph()
    if delta_args.delta_type == "lora":
        from opendelta.delta_models.lora import LoraModel
        if not delta_args.common_structure:
            if ckpt_name.startswith('distilbert'):
                delta_model = LoraModel()
                delta_model(model, modified_keys=["k_lin", "3.attention.v_lin","4.attention.v_lin"])
                delta_model.freeze_module(model, exclude=["qa_outputs", "deltas", 'bias'])
            elif ckpt_name.startswith('t5'):
                delta_model = LoraModel()
                delta_model(model, modified_keys=["SelfAttention.k", "SelfAttention.v"])
                delta_model.freeze_module(model, exclude=["deltas"])
            else:
                raise NotImplementedError
        else:
            delta_model = LoraModel(common_structure=True, structure_mapping=Mappings[model_args.model_name])
            delta_model(model, modified_keys=["attn.k", "attn.v"])
            delta_model.freeze_module(model, exclude=["qa_outputs", "deltas"])

    elif delta_args.delta_type == "bias":
        from opendelta.delta_models.bias import BiasModel
        if not delta_args.common_structure:
            if ckpt_name.startswith('distilbert'):
                delta_model = BiasModel()
                delta_model(model, modified_keys=["attention", "ffn"])
                delta_model.freeze_module(model, exclude=["qa_outputs", "deltas", "0.sa_layer_norm", "bias"])
            else:
                raise NotImplementedError
        else:
            delta_model = LoraModel(common_structure=True, structure_mapping=Mappings[model_args.model_name])
            delta_model(model, modified_keys=["attn", "ff"])
            delta_model.freeze_module(model, exclude=["qa_outputs", "deltas"])


    elif delta_args.delta_type == "prefix":
        from opendelta.delta_models.prefix import PrefixModel
        if not delta_args.common_structure:
            if  ckpt_name.startswith('distilbert'):
                delta_model = PrefixModel()
                delta_model(model, modified_keys=["attention"])
                delta_model.freeze_module(model, exclude=["qa_outputs", "deltas", "q_lin"])
            elif ckpt_name.startswith('t5'):
                delta_model = PrefixModel()
                delta_model(model, modified_keys=["SelfAttention"])
                delta_model.freeze_module(model, exclude=["qa_outputs", "deltas"])
            elif ckpt_name.startswith('gpt2'):
                delta_model = PrefixModel()
                delta_model(model, modified_keys=["attn"])
                delta_model.freeze_module(model, exclude=["qa_outputs", "deltas"])
            elif ckpt_name.startswith('bart'):
                delta_model = PrefixModel()
                delta_model(model, modified_keys=["self_attn"])
                delta_model.freeze_module(model, exclude=["qa_outputs", "deltas"])
            else:
                raise NotImplementedError
        else:
            print("use common_structure")
            delta_model = PrefixModel(common_structure=True, structure_mapping=Mappings[model_args.model_name], reparameterize=False)
            delta_model(model, modified_keys=[r"[0-7]\.attn"], is_regex=True)
            delta_model.freeze_module(model, exclude=["qa_outputs", "deltas"])

        

    elif delta_args.delta_type == "adapter":
        from opendelta.delta_models.adapter1 import AdapterModel
        if not delta_args.common_structure:
            if  ckpt_name.startswith('distilbert'):
                delta_model = AdapterModel()
                delta_model(model, modified_keys=[r"attention", "ffn"], registration_name="deltas")
                delta_model.freeze_module(model, exclude=["qa_outputs", "deltas"])
            elif ckpt_name.startswith('bart'):
                delta_model = AdapterModel()
                delta_model(model, modified_keys=[r"self_attn", "fc2"], registration_name="deltas")
                delta_model.freeze_module(model, exclude=["qa_outputs", "deltas"])
            elif ckpt_name.startswith('gpt2'):
                delta_model = AdapterModel()
                delta_model(model, modified_keys=["attn", "mlp"], registration_name="deltas")
                delta_model.freeze_module(model, exclude=["qa_outputs", "deltas"])
            elif ckpt_name.startswith('t5'):
                delta_model = AdapterModel()
                delta_model(model, modified_keys=["SelfAttention", "DenseReluDense"], registration_name="deltas")
                delta_model.freeze_module(model, exclude=["qa_outputs", "deltas"]) 
            else:
                raise NotImplementedError
        else:
            delta_model = AdapterModel(common_structure=True, structure_mapping=Mappings[model_args.model_name])
            delta_model(model, modified_keys=[r"attn", "ff"], registration_name="deltas" )
            delta_model.freeze_module(model, exclude=["qa_outputs", "deltas"])
            

    vis2 = Visualization(model)
    vis2.structure_graph()
    print(f"Tunable params rate {delta_model.num_trainable_parameters()/delta_model.num_frozen_parameters(model)}")
    return model
