from collections import OrderedDict
import argparse
import sys
import json

# TASK_TO_METRICS = {
#                 #   "mrpc": ["accuracy", "f1"],
#                 #   "cola": ['matthews_correlation'],
#                 #   "stsb": ['pearson', 'spearmanr'],
#                 #   'sst2': ['accuracy'],
#                 #   "mnli": ["accuracy"],
#                 #   "mnli_mismatched": ["accuracy"],
#                 #   "mnli_matched": ["accuracy"],
#                 #   "qnli": ["accuracy"],
#                 #   "rte": ["accuracy"],
#                 #   "wnli": ["accuracy"],
#                 #   "qqp": ["accuracy", "f1"],

#                   "superglue-boolq": ["accuracy"],
#                   "superglue-rte": ["accuracy"],
#                   "superglue-cb": ["f1_multiclass", "accuracy"],
#                   "superglue-copa": ["accuracy"],
#                   "superglue-multirc": ["f1", "em"],
#                   "superglue-wic": ["accuracy"],
#                   "superglue-wsc.fixed": ["accuracy"],
#                   "superglue-record": ["f1", "em"]
#          }
# dataset_names = list(TASK_TO_METRICS.keys())
import os
filenames = [x for x in os.listdir("../lora") if x.endswith(".json")]
# Notice! some file shouldn't be changed.
for filename in filenames:
    baseconfig = json.load(open(f"../lora/{filename}"))
    baseconfig['delta_type'] = "prefix"
    dataset_name = filename[len("lora_"):-len(".json")]
    baseconfig["output_dir"] = "/".join(baseconfig["output_dir"].split("/")[:-2]+["prefix",dataset_name])
    json.dump(baseconfig, open(f"prefix_{dataset_name}.json",'w'), indent=4,sort_keys=True)

