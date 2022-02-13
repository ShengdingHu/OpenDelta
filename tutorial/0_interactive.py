from transformers import BertForMaskedLM
model = BertForMaskedLM.from_pretrained("bert-base-cased")
# suppose we load BERT

from opendelta import LoraModel
delta_model = LoraModel(backbone_model=model, interactive_modify=True)
# This will visualize the backbone after modification and other information.

delta_model.freeze_module(exclude=["deltas", "layernorm_embedding"], set_state_dict=True)
# The set_state_dict=True will tell the method to change the state_dict of the backbone_model to maintaining only the trainable parts.

from opendelta import Visualization
print("after modify")
Visualization(model).structure_graph()
