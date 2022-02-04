# Text classification with OpenDelta
This repository contains the examples that uses OpenDelta to do text-classification in a traditional classification mode, i.e., with a classification head on top of the language model. Almost all of the training pipeline codes remain the same, except for some minimum changes to insert delta models onto the backbone model. 


## Generating the json configuration file

```
python config_gen.py --job $job_name

```
The available job configuration (e.g., `--job lora_roberta-base`) can be seen from `config_gen.py`. You can also
create your only configuration.


## Run the code

```
python run_glue.py configs/$job_name/$dataset.json
```



## Link to the original training scripts
This example repo is based on the [huggingface text-classification example](https://github.com/huggingface/transformers/tree/master/examples/pytorch/text-classification). Thanks to the authors of the original repo. 
