# OpenDelta
Implementation of various parameter efficient methods.

### Execution Tested Combination

(tested using testing/testing.py)

|   | Lora  | Bias<br>Tuning  |  Prefix<br> Tuning | Adapter<br>Houstbly | Adapter<br>Preffier| Adapter<br>Drop|  Adapater<br> Low-Rank | Compactor | Prompt <br> Tuning | Diff<br>Pruning |
|---|---|---|---|---|---|---|---|---| ---|---| 
| T5  | :white_check_mark: | :white_check_mark:  | :white_check_mark:  | :white_check_mark:  | :ballot_box_with_check:  |  :ballot_box_with_check: | :white_check_mark:| :white_check_mark: |
| GPT-2 | :ballot_box_with_check:  | :ballot_box_with_check:  |  :ballot_box_with_check:  | :ballot_box_with_check:  |  :ballot_box_with_check: |  :ballot_box_with_check: |:ballot_box_with_check:  |  :ballot_box_with_check:  | 
| BART |  :ballot_box_with_check: |   :ballot_box_with_check:|  :ballot_box_with_check:  | :ballot_box_with_check:  |  :ballot_box_with_check: | :ballot_box_with_check:  |:ballot_box_with_check:  |  :ballot_box_with_check:  | 
| DistilBERT |  :ballot_box_with_check: | :ballot_box_with_check:  | :ballot_box_with_check:  | :ballot_box_with_check:  |   :ballot_box_with_check: | :ballot_box_with_check:  |:ballot_box_with_check:  |  :ballot_box_with_check:  | 
| RoBERTa |   :white_check_mark: | :ballot_box_with_check:  | :ballot_box_with_check:  | :ballot_box_with_check:  |   :ballot_box_with_check: | :ballot_box_with_check:  |:ballot_box_with_check:  |  :ballot_box_with_check:  | 
| BERT |   :ballot_box_with_check: | :ballot_box_with_check:  | :ballot_box_with_check:  | :ballot_box_with_check:  |   :ballot_box_with_check: | :ballot_box_with_check:  |:ballot_box_with_check:  |  :ballot_box_with_check:  | 
| T5-3b |   :ballot_box_with_check: | :ballot_box_with_check:  | :ballot_box_with_check:  | :ballot_box_with_check:  |   :ballot_box_with_check: | :ballot_box_with_check:  |:ballot_box_with_check:  |  :ballot_box_with_check:  | 
| Deberta-v2 |   :ballot_box_with_check: | :ballot_box_with_check:  | :ballot_box_with_check:  | :ballot_box_with_check:  |   :ballot_box_with_check: | :ballot_box_with_check:  |:ballot_box_with_check:  |  :ballot_box_with_check:  | 
| CTRL |   :ballot_box_with_check: | :ballot_box_with_check:  | :ballot_box_with_check:  | :ballot_box_with_check:  |   :ballot_box_with_check: | :ballot_box_with_check:  |:ballot_box_with_check:  |  :ballot_box_with_check:  | 
| ViT |   :ballot_box_with_check: |   |   |   |    |   |  |    | 


### Performance Checked Combination

Google sheet [here](https://docs.google.com/spreadsheets/d/1BIVa8ocAPga-u7rBOXLYaTfaJSjI1dWfwohmLjmFDrY/edit?usp=sharing)


### TODOs

- [ ] Compactor `shared_phm_rule=True` not supported.
- [ ] delta_models/adapters folder need to be remove with no harms.
- [ ] Preffier Adatper
- [ ] Prompt Tuning
- [ ] Prefix Generation Bug

