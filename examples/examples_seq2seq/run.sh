files=(cola mnli mrpc qnli qqp rte sst2 stsb superglue-boolq superglue-cb superglue-copa superglue-multirc superglue-record superglue-wic superglue-wsc.fixed)
for ((i=$1; i<=$2; i++))
do
    dataset=${files[i]}
    echo "id$i:$dataset"
    TOKENIZERS_PARALLELISM=false python run_seq2seq.py configs/$3_t5-base/$dataset.json
done 