from collections import OrderedDict
import argparse
import sys
import json

CONFIGS = {    # Epoch batchsize maxseqlength
                  "mrpc": [30, 16, 512],
                  "cola": [80, 32, 512],
                  "stsb": [40, 16, 512],
                  'sst2': [60, 16, 512],
                  "mnli": [30, 16, 256],
                  "mnli_mismatched": [30, 16 , 512],
                  "mnli_matched": [30, 16, 512],
                  "qnli": [30, 32, 256],
                  "rte": [80, 32, 512],
                  "wnli": [30, 32, 512],
                  "qqp": [25, 16, 512],

                #   "superglue-boolq": ["accuracy"],
                #   "superglue-rte": ["accuracy"],
                #   "superglue-cb": ["f1_multiclass", "accuracy"],
                #   "superglue-copa": ["accuracy"],
                #   "superglue-multirc": ["f1", "em"],
                #   "superglue-wic": ["accuracy"],
                #   "superglue-wsc.fixed": ["accuracy"],
                #   "superglue-record": ["f1", "em"]
         }
# dataset_names = list(TASK_TO_METRICS.keys())
import os
filenames = [x for x in os.listdir(".") if x.endswith(".json")]
# Notice! some file shouldn't be changed.
for filename in filenames:
    baseconfig = json.load(open(f"{filename}"))
    baseconfig["model_name_or_path"] = "../../../../plm_cache/roberta-base"
    baseconfig["tokenizer_name"] = "../../../../plm_cache/roberta-base"
    baseconfig["model_name"] = "roberta"
    baseconfig["warmup_ratio"] = 0.06
    baseconfig["lora_alpha"] = 8
    baseconfig["lora_rank"] = 8
    baseconfig["delta_lr"] = 5e-4
    baseconfig["weight_decay"] = 0.1
    try:
      del baseconfig["metric_for_best_model"]
    except KeyError:
      pass
    dataset_name = filename[len("lora_"):-len(".json")]
    baseconfig["num_train_epochs"] = CONFIGS[dataset_name][0] if dataset_name in CONFIGS else 40
    baseconfig["per_device_train_batch_size"] = CONFIGS[dataset_name][1] if dataset_name in CONFIGS else 16
    baseconfig["max_source_length"] = CONFIGS[dataset_name][2]  if dataset_name in CONFIGS else 512
    baseconfig["output_dir"] = f"outputs/lora/roberta-base/{dataset_name}"
    json.dump(baseconfig, open(f"lora_{dataset_name}.json",'w'), indent=4,sort_keys=True)

