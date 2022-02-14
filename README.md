<div align="center">


<img src="https://s4.ax1x.com/2022/02/14/Hy7lAf.png" width="350px">

**An Open-Source Framework for Paramter Efficient Tuning.**

------

<p align="center">
  <a href="#Overview">Overview</a> •
  <a href="#installation">Installation</a> •
  <a href="#Supported-Models">Supported Models</a> •
  <a href="https://opendelta.readthedocs.io/">Docs</a> • 
  <a href="https://docs.google.com/spreadsheets/d/1BIVa8ocAPga-u7rBOXLYaTfaJSjI1dWfwohmLjmFDrY/edit?usp=sharing">Performance</a> •


</p>

</div>

![version](https://img.shields.io/badge/version-v0.1.0-blue)

## Overview

OpenDelta is a toolkit for parameter efficient methods (we dub it as *delta tuning*), by which users could flexibly assign (or add) a small amount parameters to update while keeping the most paramters frozen. 

## Installation

### Using Pip

Our repo is tested on Python 3.6+ and PyTorch 1.8.1+, install OpenDelta using pip as follows:

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
```
If you won't modify the code, run
```
python setup.py install
```

Instead, if you want to modify the code, run
```
python setup.py develop
```



### Supported Models
We verify our 

(tested using testing/testing.py)

|            | Lora                    | Bias<br>Tuning          | Prefix<br> Tuning       | Adapter<br>Houstbly     | Adapter<br>Preffier     | Adapter<br>Drop         | Adapater<br> Low-Rank   | Compactor               | Prompt <br> Tuning | Diff<br>Pruning |
| ---------- | ----------------------- | ----------------------- | ----------------------- | ----------------------- | ----------------------- | ----------------------- | ----------------------- | ----------------------- | ------------------ | --------------- |
| T5         |  ✅                     | ✅                       | ✅                       | :white_check_mark:      | :ballot_box_with_check: | :ballot_box_with_check: | :white_check_mark:      | :white_check_mark:      |                    |                 |
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



