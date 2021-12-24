from collections import OrderedDict
import argparse
import sys
import json

TASK_TO_METRICS = {
                #   "mrpc": ["accuracy", "f1"],
                #   "cola": ['matthews_correlation'],
                #   "stsb": ['pearson', 'spearmanr'],
                #   'sst2': ['accuracy'],
                #   "mnli": ["accuracy"],
                #   "mnli_mismatched": ["accuracy"],
                #   "mnli_matched": ["accuracy"],
                #   "qnli": ["accuracy"],
                #   "rte": ["accuracy"],
                #   "wnli": ["accuracy"],
                #   "qqp": ["accuracy", "f1"],

                  "superglue-boolq": ["accuracy"],
                  "superglue-rte": ["accuracy"],
                  "superglue-cb": ["f1_multiclass", "accuracy"],
                  "superglue-copa": ["accuracy"],
                  "superglue-multirc": ["f1", "em"],
                  "superglue-wic": ["accuracy"],
                  "superglue-wsc.fixed": ["accuracy"],
                  "superglue-record": ["f1", "em"]
         }
dataset_names = list(TASK_TO_METRICS.keys())
for dataset_name in dataset_names:
    baseconfig = json.load(open("baseline_superglue.json"))
    baseconfig["task_name"] = dataset_name
    baseconfig["eval_dataset_name"] = dataset_name
    baseconfig["test_dataset_name"] = dataset_name
    baseconfig["output_dir"] = baseconfig["output_dir"]+dataset_name
    json.dump(baseconfig, open(f"baseline_{dataset_name}.json",'w'), indent=4,sort_keys=True)

