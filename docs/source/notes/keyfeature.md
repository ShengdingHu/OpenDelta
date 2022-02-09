(keyfeature)=
# Philosophy and Key Features

:::{admonition} Plug-and-play Design.
:class: tip

Existing open-source project to propogate this **``delta-tuning''** paradigm includes
<a href="https://adapterhub.ml">AdapterHub</a>, which copies the transformers code base and modify on it, which makes it unintuitive to transfer from a normal code base to a delta-tuning ones.

OpenDelta approaches this problem via a **true plug-and-play** fashion to the PLMs. To migrate from a full-model finetuning training scripts to a delta tuning training scripts, you **DO NOT**  need to change the backbone bone model code base to an adapted code base.
:::


Here is how we achieve it.

<img src="../imgs/pointing-right-finger.png" height="30px"> **Read through it will also help you to implement your own delta models in a sustainable way.**

(namebasedaddr)=
## 1. Name-based submodule addressing.
We locate the submodules that we want to apply a delta layer via name-based addressing.

In pytorch fashion, a submodule can be accessed from a root model via 'dot' addressing. For example, we define a toy language model

```python
import torch.nn as nn
class MyNet1(nn.Module):
    def __init__(self,):
        super().__init__()
        self.name_a = nn.Linear(5,5)
    def forward(self, hiddens):
        return self.name_a(hiddens)

class MyNet2(nn.Module):
    def __init__(self,):
        super().__init__()
        self.embedding = nn.Embedding(10,5)
        self.name_b = nn.Sequential(MyNet1(), MyNet1())
    def forward(self, input_ids):
        hiddens = self.embedding(input_ids)
        return self.name_b(hiddens)
        
root = MyNet2()
print(root.name_b[0].name_a)
# Linear(in_features=5, out_features=5, bias=True)
```

We can visualize the model (For details, see [visualization](visualization))

```python
from opendelta import Visualization
Visualization(root).structure_graph()
```

````{collapse} <span style="color:rgb(141, 99, 224);font-weight:bold;font-style:italic">Click to view output</span>
```{figure} ../imgs/name_based_addressing.png
---
width: 500px
name: name_based_addressing
---
```
````

In this case, string `"name_b.0.name_a"` will be the name to address the submodule from the root model. 

Thus when applying a delta model to this toy net.

```
from opendelta import AdapterModel
AdapterModel(backbone_model=root, modified_modules=['name_b.0.name_a'])
Visualization(root).structure_graph()
```

````{collapse} <span style="color:rgb(141, 99, 224);font-weight:bold;font-style:italic">Click to view output</span>
```{figure} ../imgs/toy-delta.png
---
width: 500px
name: toy-delta
---
```
````

### Makes addressing easier.

Handcrafting the full names of submodules can be frustrating. We made some simplifications

1. End-matching Rules.
    OpenDelta will take every modules that 
    **ends with** the provided name suffix as the modification [target module](target_module). 
    :::{admonition} Example
    :class: tip
    Taking DistilBert with an classifier on top as an example:
    - set to `["0.attention.out_lin"]` will add delta modules to the attention output of distilbert's 
    ayer 0, i.e., `distilbert.transformer.layer.0.attention.out_lin`.
    - set to `["attention.out_lin"]` will add the delta modules in every layer's :string:`attention.out_lin`. 
    :::


2. Regular Expression.
 <img src="../imgs/todo-icon.jpeg" height="30px"> Unite test and Doc later.

3. Interactive Selection.
 <img src="../imgs/todo-icon.jpeg" height="30px"> Unite test and Doc later

## 2. Three basic submodule-level delta operations.
We use three key functions to achieve the modifications to the backbone model outside the backbone model's code.

1. **unfreeze some paramters**

   Some delta models will unfreeze a part of the model parameters and freeze other parts of the model, e.g. [BitFit](https://arxiv.org/abs/2106.10199). For these methods, just use [freeze_module](opendelta.basemodel.DeltaBase.freeze_module) method and pass the delta parts into `exclude`.
   
2. **replace an module**

   Some delta models will replace a part of the model with a delta model, i.e., the hidden states will no longer go through the original submodules. This includes [Lora](https://arxiv.org/abs/2106.09685).
   For these methods, we have an [update_module](opendelta.basemodel.DeltaBase.replace_module) interface.

3. **insertion to the backbone**

   - **sequential insertion**
   
    Most adapter model insert a new adapter layer after/before the original transformers blocks. For these methods, insert the adapter's forward function after/before the original layer's forward function using [insert_sequential_module](opendelta.basemodel.DeltaBase.insert_sequential_module) interface. 
   - **parallel insertion**
   
    Adapters can also be used in a parallel fashion (see [Paper](https://arxiv.org/abs/2110.04366)).
    For these methods, use [insert_parallel_module](opendelta.basemodel.DeltaBase.insert_parrellel_module) interface.


:::{admonition} Doc-preserving Insertion
:class: note
In the insertion operations, the replaced forward function will inherit the doc strings of the original functions. 
:::

## 3. Pseudo input to initialize.
Some delta models, especially the ones that is newly introduced into the backbone, will need to determine the parameters' shape. To get the shape, we pass a pseudo input to the backbone model and determine the shape of each delta layer according to the need of smooth tensor flow. 

:::{admonition} Pseudo Input
:class: warning
Most models in [Huggingface Transformers](https://huggingface.co/docs/transformers/index) have an attribute [dummy_inputs](https://github.com/huggingface/transformers/blob/v4.16.2/src/transformers/modeling_utils.py#L464). This will create a nonsensical input with the correct format to pass into the model's forward function.

For the models that doesn't inherit/implement this attributes, we assume the pseudo input to the model is something like `input_id`, i.e., an integer tensor.
```python
pseudo_input = torch.tensor([[0,0,0]])
# or 
pseudo_input = torch.tensor([0,0,0])
```
<img src="../imgs/todo-icon.jpeg" height="30px"> We will add interface to allow more pseudo input in the future.
:::





