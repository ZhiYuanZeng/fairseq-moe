# SCoMoE
## Implementation of SCoMoE
The path of the implementation of SCoMoE is `user_dir/fast_moe/`

The core implementation
- SCoMoE-seq: `./user_dir/fast_moe/moe.py (EffecientMoeLayer.scomoe_seq)`
- SCoMoE-dmodel: `./user_dir/fast_moe/moe.py (EffecientMoeLayer.scomoe_dmodel)`
- token clustering: `./user_dir/fast_moe/moe.py (EffecientMoeLayer.token_cluering_moe)`
- differentible sorting: `./user_dir/fast_moe/gates.py (BalanceGate)`

## Requirements
- python>=3.7
- pytorch>=1.09
- Fairseq (install from source code)
- sacrebleu < 2.0
- slurm (for running multi-node training/inference)

## installment
`pip insall -e ./`

`python setup.py build_ext --inplace`

## Data processing
Process:
1. BPE training with sentencepiece
2. BPE apply
3. Binarizing the data with `fairseq-preprocess`

OPUS-100: download data from opus website. (the scripts for preprocessing of the OPUS-100 data is in `./preprocess_scripts/multilingual_preprocess.sh` )
WMT17-En-Fr: download and clean data with `examples/translation/prepare-wmt14en2fr.sh`

## Model training
The scripts below are all for training Gshard. To train SCoMoE, just append the `${scomoe_seq_args}` or `${scomoe_feat_args}` at the end of command.


dummy train (training with dummy data): 
- `train_scripts/scomoe/dummy_train_base_model.sh`
- `train_scripts/scomoe/dummy_train_large_model.sh`

train on WMT17-en-fr: 
- base models: `bash train_scripts/scomoe/train_base_models_on_wmt-en-fr.sh`
- large models: `sbatch train_scripts/scomoe/train_large_model_on_wmt-en-fr.sh`

train on OPUS-100:


### Explaination for important args of SCoMoE:
args for scomoe-seq
- `--arch scomoe`: scomoe
- `--scomoe-type seq`: scomoe-seq
- `--ratio1`: the ratio for intra-accelerator communication ($\alpha$)
- `--ratio2`: the ratio for intra-node communication ($\beta$)
- `--token-cluster`: whether do token clustering
- `--temperature`: temperature for softmax in token clustering

args for scomoe-feat
- `--arch scomoe`: scomoe
- `--scomoe-type feat`: scomoe-feat
- `--ratio1`: the ratio for intra-accelerator communication ($\alpha$)
- `--ratio2`: the ratio for intra-node cmmunication ($\beta$)

## Model Evaluation
Evauation on WMT17-En-Fr: 
- `bash eval_scripts/eval_on_wmt-en-fr.sh --save_dir ${ckpt_path} -subset {valid|test} -capacity_factor ${eval_capacity_factor} --r1 ${ratio1} --r2 ${ratio2}$`
- `bash eval_scripts/eval_large_on_wmt-en-fr.sh --save_dir ${ckpt_path} -subset {valid|test} -capacity_factor ${eval_capacity_factor} --r1 ${ratio1} --r2 ${ratio2}$`

Evauation on OPUS-100: 
- `bash eval_on_opus.sh --save_dir ${ckpt_path} -subset {valid|test} -capacity_factor ${eval_capacity_factor} --r1 ${ratio1} --r2 ${ratio2}$`
- `bash eval_large_on_opus.sh --save_dir ${ckpt_path} -subset {valid|test} -capacity_factor ${eval_capacity_factor} --r1 ${ratio1} --r2 ${ratio2}$`


`ratio1` and `ratio2` are only necessary for the evaluation of Gshard. For SCoMoE-seq, it is better to set `ratio1=0` and `ratio2=0`. While for SCoMoE-Feat, they should be the same values as those at training.

`eval_capacity_factor` has a great influence on the translation performance. For evaluation on OPUS-100, `eval_capacity_factor` should be set to at least 0.5. As for the evaluation on OPUS-100, setting `eval_capacity_factor=0.25` is enough.