# There is a recent trend to unify all tasks (such as classification) into tasks generation.
# In fact, unifying the tasks into text generation can be neatly conducted using prompt. 
# In OpenPrompt, we provide a GenerationVerbalizer for this utility.
# Here we go!

from openprompt.pipeline_base import PromptForGeneration
from openprompt.prompts.generation_verbalizer import GenerationVerbalizer
from tokenizers import PreTokenizedString
from tqdm import tqdm

import torch
from openprompt.data_utils.utils import InputExample
import argparse
import numpy as np
import loralib as lora

from openprompt import PromptDataLoader
from openprompt.prompts import ManualVerbalizer
from openprompt.prompts import ManualTemplate
import time
import os
import re
from openprompt.utils.crossfit_metrics import evaluate as crossfit_evaluate
from openprompt.plms import load_plm
import random
from openprompt.utils.reproduciblity import set_seed
from transformers import  AdamW, get_linear_schedule_with_warmup,get_constant_schedule_with_warmup  # use AdamW is a standard practice for transformer 
from transformers.optimization import Adafactor, AdafactorSchedule  # use Adafactor is the default setting for T5
from openprompt.data_utils.data_sampler import FewShotSampler
from openprompt.data_utils import PROCESSORS
from yacs.config import CfgNode
from opendelta.delta_models.bias_t5 import BiasModel


def get_dataset_specific_config(args):  
    config = CfgNode(new_allowed=True)
    dataset = {}
    config.dataset_decoder_max_length = 10
    config.max_seq_l = 320 # this should be specified according to the running GPU's capacity 
    config.batchsize_t = 4
    config.batchsize_e = 4
    config.gradient_accumulation_steps = 4
    config.model_parallelize = False

    Processor = PROCESSORS[f"super_glue.{args.dataset}"]
    dataset['train'] = Processor().get_train_examples(args.data_dir)
    dataset['validation'] = Processor().get_dev_examples(args.data_dir)
    dataset['test'] = Processor().get_test_examples(args.data_dir)
    config.class_labels =Processor().get_labels()
    
    if args.dataset == "cb":
        config.textual_template =  ' hypothesis: {"placeholder":"text_b","post_processing": lambda x:x+"."} premise: {"placeholder":"text_a"} {"mask"}'
        config.verbalizer_text = [{"text": "entailment"}, {"text": "contradiction"}, {"text": "neutral"}]
    elif args.dataset == "boolq":
        config.verbalizer_text = [{"text":"No"}, {"text":"Yes"}]
        config.textual_template = ' hypothesis: {"placeholder":"text_b", "shortenable":False, "post_processing": lambda x:x+"."} premise: {"placeholder":"text_a"} {"mask"}'
    elif args.dataset == 'rte':
        config.textual_template = ' sentence1: {"placeholder":"text_a"} sentence2: {"placeholder":"text_b"} {"mask"}'
        config.verbalizer_text = [{"text": "entailment"}, {"text": "contradiction"}]
    elif args.dataset == 'wsc':
        config.textual_template = ' {"placeholder":"text_a"} "{"meta":"span2_text"}" refers to "{"meta":"span1_text"}" or another word ? {"mask"}'
        config.verbalizer_text = [{"text": "Another word"}, {"meta": "span1_text"}]
    elif args.dataset == 'wic':
        config.textual_template = ' sentence1: {"placeholder":"text_a"} sentence2: {"placeholder":"text_b"} word: {"meta":"word", "shortenable": False} {"mask"}'
        config.verbalizer_text = [{"text": "No"}, {"text": "Yes"}]
    elif args.dataset == 'multirc':
        config.textual_template = ' question: {"placeholder":"text_b", "shortenable":False} answer: {"meta":"answer", "shortenable":False, "post_processing": lambda x:x+"."} paragraph: {"placeholder":"text_a"} {"mask"}'
        config.verbalizer_text = [{"text": "No"}, {"text": "Yes"}]
    elif args.dataset == "copa":
        config.verbalizer_text  = [{"meta":"choice1"}, {"meta":"choice2"}]
        config.textual_template =  ' choice1: {"meta":"choice1"} choice2: {"meta":"choice2"} premise: {"placeholder":"text_a"} question: {"meta":"question"} {"mask"}'
        config.dataset_decoder_max_length = 50
    elif args.dataset == "record":
        config.textual_template = 'query: {"meta":"query"} context: {"meta": "passage", "shortenable":True} entities: {"meta":"entities", "shortenable":True} {"mask"}'
        config.verbalizer_text = [{"meta": "answers"}]
        config.dataset_decoder_max_length = 20
    return dataset, config

def evaluate(prompt_model, dataloader):
    predictions = []
    ground_truths = []

    for step, inputs in enumerate(dataloader):
        inputs = inputs.cuda()
        _, output_sentence = prompt_model.generate(inputs, **generation_arguments, verbose=False)
        predictions.extend(output_sentence)
        ground_truths.extend(inputs['tgt_text'])
    assert len(predictions)==len(ground_truths), (len(predictions), len(ground_truths))
    predictions = [prediction.strip() for prediction in predictions]
    ground_truths = [ground_truth.strip() for ground_truth in ground_truths]
    # shown one example
    print(f"predictions {predictions[0]}, ground_truths {ground_truths[0]}")
    score =  crossfit_evaluate(predictions, ground_truths, metric="ACC")
    return score

DIRBASE = "/home/hushengding/"
parser = argparse.ArgumentParser("")
parser.add_argument("--shot", type=int, default=-1)
parser.add_argument("--seed", type=int, default=144)
parser.add_argument("--plm_eval_mode", action="store_true", help="whether to turn off the dropout in the freezed model. Set to true to turn off.")
parser.add_argument("--model", type=str, default='t5-lm', help="We test both t5 and t5-lm in this scripts, the corresponding tokenizerwrapper will be automatically loaded.")
parser.add_argument("--model_name_or_path", default=f'{DIRBASE}/plm_cache/t5-base-lm-adapt/')
parser.add_argument("--project_root", default=f"{DIRBASE}/OpenDelta_dev/OpenDelta/", help="The project root in the file system, i.e. the absolute path of OpenPrompt")
parser.add_argument("--template_id", type=int)
parser.add_argument("--verbalizer_id", type=int)
parser.add_argument("--data_dir", type=str, default=f"{DIRBASE}/huggingface_datasets/saved_to_disk/") # sometimes, huggingface datasets can not be automatically downloaded due to network issue, please refer to 0_basic.py line 15 for solutions. 
parser.add_argument("--dataset",type=str)
parser.add_argument("--result_file", type=str, default="../sfs_out/results.txt")
parser.add_argument("--max_steps", default=20000, type=int)
parser.add_argument("--delta_lr", type=float, default=3e-5)
# dias specific 


parser.add_argument("--warmup_step", type=int, default=500)
parser.add_argument("--eval_every_steps", type=int, default=500)
parser.add_argument("--optimizer", type=str, default="Adafactor")
args = parser.parse_args()
args.result_file = os.path.join(args.project_root, args.result_file)

content_write = "="*20+"\n"
content_write += f"dataset {args.dataset}\t"
content_write += f"temp {args.template_id}\t"
content_write += f"verb {args.verbalizer_id}\t"
content_write += f"model {args.model}\t"
content_write += f"seed {args.seed}\t"
content_write += f"shot {args.shot}\t"
content_write += f"plm_eval_mode {args.plm_eval_mode}\t"
content_write += f"eval_every_steps {args.eval_every_steps}\t"
content_write += f"delta_lr {args.delta_lr}\t"
content_write += f"optimizer {args.optimizer}\t"
content_write += f"warmup_step {args.warmup_step}\t"
content_write += "\n"
print(content_write)
# set seed
this_run_unicode = str(random.randint(0, 1e10))
set_seed(args.seed)

# load dataset
dataset, dconfig = get_dataset_specific_config(args)

# load plm and set it to eval mode
plm, tokenizer, model_config, WrapperClass = load_plm(args.model, args.model_name_or_path)

plm = plm.cuda()
mydeltamodel = BiasModel()
plm = mydeltamodel(plm)
mydeltamodel.cuda()

# check trainable params
print("trainable parameter name", mydeltamodel.trainable_parameters_names, len(mydeltamodel.trainable_parameters_names))
plm_trainable_params = [n for n,p in plm.named_parameters() if p.requires_grad]
print("plm trainable parameter name", plm_trainable_params, len(plm_trainable_params))

# pipeline related 
mytemplate = ManualTemplate(tokenizer=tokenizer, text=dconfig.textual_template)
myverbalizer = GenerationVerbalizer(tokenizer, classes=dconfig.class_labels, is_rule=True, label_words=dconfig.verbalizer_text)
prompt_model = PromptForGeneration(plm=plm,template=mytemplate, plm_eval_mode=args.plm_eval_mode).cuda()

if dconfig.model_parallelize:
    prompt_model.parallelize()

# currenlty still use prompt dataloader
train_dataloader = PromptDataLoader(dataset=dataset["train"], template=mytemplate, verbalizer=myverbalizer, tokenizer=tokenizer, # be sure to add verbalizer 
    tokenizer_wrapper_class=WrapperClass, max_seq_length=dconfig.max_seq_l, decoder_max_length=dconfig.dataset_decoder_max_length,  # be sure to use larger decoder_max_length for teacher forcing.
    batch_size=dconfig.batchsize_t,shuffle=True, teacher_forcing=True, predict_eos_token=True,  # be sure to use teacher_forcing and predict_eos_token=True
    truncate_method="tail")
validation_dataloader = PromptDataLoader(dataset=dataset["validation"], template=mytemplate, verbalizer=myverbalizer, tokenizer=tokenizer, 
    tokenizer_wrapper_class=WrapperClass, max_seq_length=dconfig.max_seq_l, decoder_max_length=3, 
    batch_size=dconfig.batchsize_e,shuffle=False, teacher_forcing=False, predict_eos_token=False, # predict_eos_token=True or False are both ok 
    truncate_method="tail")
test_dataloader = PromptDataLoader(dataset=dataset["test"], template=mytemplate, verbalizer=myverbalizer, tokenizer=tokenizer, 
    tokenizer_wrapper_class=WrapperClass, max_seq_length=dconfig.max_seq_l, decoder_max_length=3, 
    batch_size=dconfig.batchsize_e,shuffle=False, teacher_forcing=False, predict_eos_token=False,
    truncate_method="tail")
print("truncate rate: {}".format(test_dataloader.tokenizer_wrapper.truncate_rate), flush=True)


generation_arguments = {
    "max_length": dconfig.dataset_decoder_max_length,
}
loss_func = torch.nn.CrossEntropyLoss()
tot_step = args.max_steps

# optimizer related 
optimizer_grouped_parameters = []
optimizer_grouped_parameters.extend([
    {'params': mydeltamodel.trainable_parameters, 'weight_decay': 0.01, 'lr': args.delta_lr},
])
if args.optimizer.lower() == "adafactor":
    optimizer = Adafactor(optimizer_grouped_parameters,  
                                relative_step=False,
                                scale_parameter=False,
                                warmup_init=False)  # when lr is 0.3, it is the same as the configuration of https://arxiv.org/abs/2104.08691
    scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_step) # when num_warmup_steps is 0, it is the same as the configuration of https://arxiv.org/abs/2104.08691
elif args.optimizer.lower() == "adamw":
    optimizer = AdamW(optimizer_grouped_parameters) # usually lr = 0.5
    scheduler = get_linear_schedule_with_warmup(
                    optimizer, 
                    num_warmup_steps=args.warmup_step, num_training_steps=tot_step) # usually num_warmup_steps is 500

plm_total_params = sum(p.numel() for p in plm.parameters())

def get_num_optimized_parameters(opt):
    pnum_tot = 0
    for pg in opt.param_groups:
        params = pg['params']
        for param in params:
            pnum_tot += param.numel()
    return pnum_tot
num_params = get_num_optimized_parameters(optimizer)
tune_rate = num_params/plm_total_params
print(f"Trainable parameters num: {mydeltamodel.num_trainable_parameters} Num_params in optimizer: {num_params}, total num of params in plm: {plm_total_params}, tune rate: {tune_rate*100:.4}%")


content_write += f"Num_params\t{mydeltamodel.num_trainable_parameters}"
content_write += f"tune_rate \t{tune_rate}"

# training
tot_loss = 0
log_loss = 0
best_val_acc = 0
glb_step = 0
actual_step = 0
leave_training = False

acc_traces = []
tot_train_time = 0
pbar_update_freq = 10
prompt_model.train()

pbar = tqdm(total=tot_step, desc="Train")
for epoch in range(1000000):
    for step, inputs in enumerate(train_dataloader):
        inputs = inputs.cuda()
        tot_train_time -= time.time()
        loss = prompt_model(inputs)
        loss.backward()
        tot_loss += loss.item()
        actual_step += 1


        if actual_step % dconfig.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(prompt_model.parameters(), 1.0)
            glb_step += 1
            if glb_step % pbar_update_freq == 0:
                aveloss = (tot_loss - log_loss)/pbar_update_freq
                pbar.update(10)
                pbar.set_postfix({'loss': aveloss})
                log_loss = tot_loss

        
        
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

        tot_train_time += time.time()

        if actual_step % dconfig.gradient_accumulation_steps == 0 and glb_step >0 and glb_step % args.eval_every_steps == 0:
            val_acc = evaluate(prompt_model, validation_dataloader)
            if val_acc >= best_val_acc:
                torch.save(prompt_model.state_dict(),f"{args.project_root}/../ckpts/{this_run_unicode}.ckpt")
                best_val_acc = val_acc
            
            acc_traces.append(val_acc)
            print("Glb_step {}, val_acc {}, average time {}".format(glb_step, val_acc, tot_train_time/actual_step ), flush=True)
            prompt_model.train()

        if glb_step > args.max_steps:
            leave_training = True
            break
    
    if leave_training:
        break  
    


# a simple measure for the convergence speed.
thres99 = 0.99*best_val_acc
thres98 = 0.98*best_val_acc
thres100 = best_val_acc
step100=step98=step99=args.max_steps
for val_time, acc in enumerate(acc_traces):
    if acc>=thres98:
        step98 = min(val_time*args.eval_every_steps, step98)
        if acc>=thres99:
            step99 = min(val_time*args.eval_every_steps, step99)
            if acc>=thres100:
                step100 = min(val_time*args.eval_every_steps, step100)


content_write += f"BestValAcc:{best_val_acc}\tEndValAcc:{acc_traces[-1]}\tcritical_steps:{[step98,step99,step100]}\n"
content_write += "\n"

print(content_write)

with open(f"{args.result_file}", "a") as fout:
    fout.write(content_write)

import os
os.remove(f"../ckpts/{this_run_unicode}.ckpt")