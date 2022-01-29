# Apping OpenDelta to GLUE/SuperGLUE tasks using Seq2Seq Paradigm

```bash
cd ../
python setup_seq2seq.py develop
```
This will add `examples_seq2seq` to the environment path of the python lib.

Then run (e.g.)
```bash
deltatype="prefix"
python run_seq2seq_deltas.py configs/${deltatype}/${deltatype}_superglue-boolq.json
```

