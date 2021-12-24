# Use Examples

This repo mainly contains three types of examples (we may add more in the future) of using OpenDelta to conduct
parameter-efficient training of various tasks.

## Language Understanding on GLUE and SuperGLUE.
We provide the example in `examples_seq2seq/` folder. To make sure the relative path of the used packages are correct,
please run the following command. 

```bash
python setup_seq2seq.py develop
```
This will add `examples_seq2seq` to the environment path of the python lib.

Then run (e.g.)
```bash
python run_seq2seq_deltas.py configs/prefix/prefix_superglue-boolq.json
```

## Integration with Prompt-tuning Methods with OpenPrompt

TODO

## Other tasks

TODO
