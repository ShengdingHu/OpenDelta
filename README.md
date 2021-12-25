# OpenDelta
Implementation of various parameter efficient methods.

### Execution Tested Combination

(tested using examples/testing.py)

|   | Lora  | Bias  |  Prefix | adapter | low_rank_adapter | compactor |
|---|---|---|---|---|---|---|
| t5 | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark:  | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark:|
| gpt2 | :heavy_check_mark:  | :heavy_check_mark:  |  :heavy_check_mark: |  :heavy_check_mark: |
| bart |  :heavy_check_mark: |   :heavy_check_mark:|  :heavy_check_mark: | :heavy_check_mark:  |
| distilbert |  :heavy_check_mark: | :heavy_check_mark:  |  :heavy_check_mark: | :heavy_check_mark:  |
| roberta |   |   |   |  |
| bert |  |   |  |  |

### Performance Checked Combination

Google sheet [here](https://docs.google.com/spreadsheets/d/1BIVa8ocAPga-u7rBOXLYaTfaJSjI1dWfwohmLjmFDrY/edit?usp=sharing)


### TODOs

- [ ] Compactor `shared_phm_rule=True` not supported.
- [ ] delta_models/adapters folder need to be remove with no harms.
- [ ] Preffier Adatper
- [ ] Prompt Tuning
- [ ] Prefix Generation Bug

