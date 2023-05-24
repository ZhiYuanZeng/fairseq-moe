# fairseq-moe
## Overview
An MoE implementation based on the moe branch of fairseq.

Supported models
- gshard
- switch transformer
- scomoe: implementation of scomoe (https://openreview.net/forum?id=s-c96mSU0u5)

Optimizations based on moe branch of fairseq:
1. dist-mmap: support index sharding for training on large corpus
2. data-parallel-inference: faster inference with data-parallel
3. multilingual inference: support load model once and translate on multilingual datasets
4. lang-pair language token: support using lang-pair as encoder language token by setting `--encoder-langtok lang_pair`

## Requirements
- python>=3.7
- pytorch>=1.09
- boto3
- iopath
- Fairseq (install from source code)
- sacrebleu < 2.0
- slurm (for running multi-node training/inference)
- fairscale

## Installment
`pip install -e ./`

`python setup.py build_ext --inplace`

## Data processing
Process:
1. BPE training with sentencepiece
2. BPE apply
3. Binarizing the data with `fairseq-preprocess`

datasets:
- OPUS-100: download data from opus website. The preprocessing pipeline follows "https://github.com/cordercorder/nmt-multi" (you can find the scripts for preprocessing at `https://github.com/cordercorder/nmt-multi/blob/main/scripts/opus-100/data_process/multilingual_preprocess.sh` )

- WMT17-En-Fr: download and clean data with `examples/translation/prepare-wmt14en2fr.sh`

## Training
The scripts below are all for training Gshard. To train SCoMoE, just append the `${scomoe_seq_args}` or `${scomoe_feat_args}` at the end of command.


dummy train (training with dummy data): 
- `train_scripts/scomoe/dummy_train_base_model.sh`
- `train_scripts/scomoe/dummy_train_large_model.sh`

train on WMT17-en-fr: 
- base models: `bash train_scripts/scomoe/train_base_models_on_wmt-en-fr.sh`
- large models: `sbatch train_scripts/scomoe/train_large_model_on_wmt-en-fr.sh`

train on OPUS-100:
- base models: `bash train_scripts/scomoe/train_base_model_on_opus100.sh`
- large models: `sbatch train_scripts/scomoe/train_large_model_on_opus100.sh`


### Explaination for args of SCoMoE:
args for scomoe-seq
- `--arch scomoe`: scomoe
- `--scomoe-type seq`: scomoe-seq
- `--ratio1`: the ratio for intra-accelerator communication ($\alpha$)
- `--ratio2`: the ratio for intra-node communication ($\beta$)
- `--token-cluster`: whether to do token clustering
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


`ratio1` and `ratio2` are only necessary for the evaluation of SCoMoE. For SCoMoE-seq, it is better to set `ratio1=0` and `ratio2=0`. While for SCoMoE-Feat, they should be the same values as those at training.

`eval_capacity_factor` has a great influence on the translation performance. For evaluation on OPUS-100, `eval_capacity_factor` should be set to at least 0.5. As for the evaluation on WMT17-en-fr, setting `eval_capacity_factor=-1` is enough.

## Hyperparamter tuning
According to our experiments, ratio1 ($\alpha$) should be set to a low value, like 0.2, while ratio2 ($\beta$) should be set to a high value, like 0.5.

If the model is SCoMoE-Feat, ratio1 and ratio2 should be the power of 2, for example 0.5, 0.25, 0.125, otherwise the communication will be slow.

## Optimizations based on moe benchmark of fairseq
### dist-mmap
fairseq use `mmap` to load datasets, which loads the data stored in `.bin` file according to the data index stored in `.index` file. The data index records the position of each sentence in `.bin` file. While training, fairseq loads all `.index` file in memmory, which requires huge memory if dataset is large. To avoids memory overflow, we implement `dist-mmap`, which shard the data index on multiple process. That means each process only loads a part of data index. However, it only works for data parallel and moe parallel.

### data-parallel inference
MoE parallel is the combination of data parallel and model parallel. Both data and model parameters are sharded on multiple processed. However, the fairseq implementation forces all processes to run the same batch at inference, because the decoding of NMT is not synchronized. Some processes generate <eos> and stop the decoding, while other processes are still decoding, which can cause communication problem. Forcing all processes to run the same data is a solution, but not a good solution, because that makes the inference very slow. We solve this problem by explicitly synchronize processes, and force all processes to stop decoding at the same time.

## Acknowledgement
Thanks for [cordercorder](https://github.com/cordercorder), who helped me preprocess the opus-100 dataset and fix many bugs.