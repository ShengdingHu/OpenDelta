<div align="center">

# OpenDelta

**An Open-Source Framework for Paramter Efficient Tuning.**

------

<p align="center">
  <a href="#Overview">Overview</a> •
  <a href="#installation">Installation</a> •
  <a href="#Execution-Tested-Combination">Execution Tested Combination</a> •
  <a href="https://opendelta.readthedocs.io/">Docs</a> • 
  <a href="https://docs.google.com/spreadsheets/d/1BIVa8ocAPga-u7rBOXLYaTfaJSjI1dWfwohmLjmFDrY/edit?usp=sharing">Performance</a> •


</p>

</div>

![version](https://img.shields.io/badge/version-v0.1.0-blue)

## Overview

OpenDelta is a toolkit for parameter efficient methods (we dub it as *delta tuning*), by which users could flexibly assign (or add) a small amount parameters to update while keeping the most paramters frozen. 

## Installation

### Using Pip

Our repo is tested on Python 3.6+ and PyTorch 1.8.1+, install OpenPrompt using pip as follows:

```shell
pip install opendelta
```

To play with the latest features, you can also install OpenDelta from the source.

### Using Git

Clone the repository from github: 

```shell
git clone https://github.com/thunlp/OpenDelta.git
cd OpenDelta
pip install -r requirements.txt
python setup.py install
```

Modify the code

```
python setup.py develop
```



### Execution Tested Combination

(tested using testing/testing.py)

|            | Lora                    | Bias<br>Tuning          | Prefix<br> Tuning       | Adapter<br>Houstbly     | Adapter<br>Preffier     | Adapter<br>Drop         | Adapater<br> Low-Rank   | Compactor               | Prompt <br> Tuning | Diff<br>Pruning |
| ---------- | ----------------------- | ----------------------- | ----------------------- | ----------------------- | ----------------------- | ----------------------- | ----------------------- | ----------------------- | ------------------ | --------------- |
| T5         | ✅                       | ✅                       | ✅                       | :white_check_mark:      | :ballot_box_with_check: | :ballot_box_with_check: | :white_check_mark:      | :white_check_mark:      |                    |                 |
| GPT-2      | :ballot_box_with_check: | ☑️                      | :ballot_box_with_check: | :ballot_box_with_check: | :ballot_box_with_check: | :ballot_box_with_check: | :ballot_box_with_check: | :ballot_box_with_check: |                    |                 |
| BART       | :ballot_box_with_check: | :ballot_box_with_check: | :ballot_box_with_check: | :ballot_box_with_check: | :ballot_box_with_check: | :ballot_box_with_check: | :ballot_box_with_check: | :ballot_box_with_check: |                    |                 |
| DistilBERT | :ballot_box_with_check: | :ballot_box_with_check: | :ballot_box_with_check: | :ballot_box_with_check: | :ballot_box_with_check: | :ballot_box_with_check: | :ballot_box_with_check: | :ballot_box_with_check: |                    |                 |
| RoBERTa    | :white_check_mark:      | :ballot_box_with_check: | :ballot_box_with_check: | :ballot_box_with_check: | :ballot_box_with_check: | :ballot_box_with_check: | :ballot_box_with_check: | :ballot_box_with_check: |                    |                 |
| BERT       | :ballot_box_with_check: | :ballot_box_with_check: | :ballot_box_with_check: | :ballot_box_with_check: | :ballot_box_with_check: | :ballot_box_with_check: | :ballot_box_with_check: | :ballot_box_with_check: |                    |                 |
| T5-3b      | :ballot_box_with_check: | :ballot_box_with_check: |                         | :ballot_box_with_check: | :ballot_box_with_check: | :ballot_box_with_check: | :ballot_box_with_check: | :ballot_box_with_check: |                    |                 |
| Deberta-v2 | :ballot_box_with_check: | :ballot_box_with_check: |                         | :ballot_box_with_check: | :ballot_box_with_check: | :ballot_box_with_check: | :ballot_box_with_check: | :ballot_box_with_check: |                    |                 |
| CTRL       | :ballot_box_with_check: | :ballot_box_with_check: |                         | :ballot_box_with_check: | :ballot_box_with_check: | :ballot_box_with_check: | :ballot_box_with_check: | :ballot_box_with_check: |                    |                 |
| ViT        | :ballot_box_with_check: |                         |                         |                         |                         |                         |                         |                         |                    |                 |


### Performance Checked Combination

Google sheet [here](https://docs.google.com/spreadsheets/d/1BIVa8ocAPga-u7rBOXLYaTfaJSjI1dWfwohmLjmFDrY/edit?usp=sharing)


### TODOs

- [ ] Compactor `shared_phm_rule=True` not supported.
- [ ] delta_models/adapters folder need to be remove with no harms.
- [ ] Preffier Adatper
- [ ] Prompt Tuning
- [ ] Prefix Generation Bug

