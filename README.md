# OpenDelta
Implementation of various parameter efficient methods.

### Execution Tested Combination

(tested using examples/testing.py)

|   | Lora  | Bias  |  Prefix | adapter |
|---|---|---|---|---|
| t5 | [x] | [x] | [x]  | [x]  |
| gpt2 | [x]  | [x]  |  [x] |  [x] |
| bart |  [x] |   [x]|  [x] | [x]  |
| distilbert |  [x] | [x]  |  [x] | [x]  |
| roberta |   |   |   |  |
| bert |  |   |  |  |

### Performance Checked Combination

Google sheet [here](https://docs.google.com/spreadsheets/d/1BIVa8ocAPga-u7rBOXLYaTfaJSjI1dWfwohmLjmFDrY/edit?usp=sharing)

### TODOs

[] Compactor `shared_phm_rule=True` not supported.
[] delta_models/adapters folder need to be remove with no harms.
[] Preffier Adatper
[] Prompt Tuning
[] Prefix Generation Bug

