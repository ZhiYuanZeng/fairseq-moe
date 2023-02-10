#!/bin/bash
#SBATCH --nodes 4
#SBATCH --ntasks-per-node 8
#SBATCH --cpus-per-task 9
#SBATCH --gpus-per-node 8
#SBATCH -w 16,17,18,19

export NCCL_DEBUG=INFO
export NCCL_BUFFSIZE=1048576
python=/path/to/python
NUM_EXPERTS=$(expr ${SLURM_JOB_NUM_NODES} \* 8)
data_dir=/path/to/opus100

data_args=" ${data_dir}/many-many/main_data_bin \
            --lang-dict ${data_dir}/lang_dict.txt \
            --lang-pairs en-fr,cy-en,hu-en,en-lt,en-mg,yi-en,as-en,en-mr,uz-en,eo-en,li-en,es-en,ka-en,am-en,en-he,en-ja,nb-en,en-ku,en-cs,en-fi,si-en,en-no,en-se,az-en,en-ga,da-en,en-vi,eu-en,en-pa,ca-en,id-en,en-eu,cs-en,kn-en,te-en,en-ug,en-be,rw-en,gu-en,en-cy,en-tt,en-am,xh-en,en-nb,sv-en,sq-en,en-nn,en-bn,ha-en,en-hu,en-pl,en-ko,en-tg,en-zu,en-nl,ps-en,af-en,be-en,ga-en,mg-en,en-mt,bs-en,or-en,bn-en,en-sr,tg-en,hi-en,fr-en,se-en,en-hr,en-eo,en-de,en-it,sk-en,tt-en,is-en,km-en,en-br,nn-en,vi-en,en-ka,ne-en,en-et,ro-en,en-ha,fa-en,oc-en,en-sh,ko-en,en-yi,en-fa,it-en,no-en,en-ig,en-af,en-da,en-th,ur-en,en-pt,zu-en,ja-en,zh-en,ar-en,en-ky,fi-en,en-mk,lv-en,my-en,en-kk,ta-en,en-ca,mt-en,fy-en,en-uk,th-en,el-en,ml-en,et-en,en-my,en-es,en-sv,wa-en,en-sk,en-ro,en-oc,bg-en,en-uz,tr-en,sl-en,sh-en,de-en,en-lv,en-is,en-km,mr-en,en-hi,pa-en,en-gu,hr-en,en-tk,en-ta,pl-en,en-kn,lt-en,en-ps,ug-en,en-bg,br-en,en-ru,en-sl,en-ne,en-te,en-bs,tk-en,gl-en,en-si,en-rw,sr-en,pt-en,en-tr,ky-en,en-gd,ku-en,en-id,en-ur,en-li,uk-en,en-or,en-sq,gd-en,en-ar,en-ml,kk-en,en-el,en-zh,en-gl,en-as,ig-en,ms-en,nl-en,en-fy,en-az,he-en,en-ms,ru-en,mk-en,en-wa,en-xh \
            --encoder-langtok src \
            --decoder-langtok \
            --source-dict ${data_dir}/many-many/main_data_bin/dict.txt \
            --target-dict ${data_dir}/many-many/main_data_bin/dict.txt \
            --task translation_multi_simple_epoch"
save_args=" --save-dir /path/to/save_dir/ \
            --tensorboard-logdir /path/to/tensorboard_dir"
model_args="--arch transformer \
            --encoder-layers 12 \
            --decoder-layers 12 \
            --encoder-embed-dim 1024 \
            --decoder-embed-dim 1024 \
            --encoder-ffn-embed-dim 4096 \
            --decoder-ffn-embed-dim 4096"
moe_args="--moe-gating-use-fp32 \
        --moe-second-expert-policy all \
        --moe-normalize-expert-grad sqrt_world_size \
        --criterion moe_cross_entropy \
        --moe-gate-loss-wt 0.1 \
        --moe-gate-loss-combine-method sum \
        --moe-batch-prioritized-routing \
        --use-moe-pad-mask \
        --moe-freq 2 \
        --moe-expert-count ${NUM_EXPERTS} \
        --capacity-factor 1.25"
scomoe_seq_args="--arch scomoe \
          --user-dir ./user_dir \
          --scomoe-type seq \
          --ratio1 0.0 \
          --ratio2 0.9 \
          --token-cluster \
          --df-gate-type softmax \
          --temperature 0.5"
scomoe_feat_args="--arch scomoe \
          --user-dir ./user_dir \
          --scomoe-type feat \
          --ratio1 0.0 \
          --ratio2 0.9"

echo ${NUM_EXPERTS}
srun $python train.py \
$data_args \
--fp16 \
--distributed-port 12345 \
--share-all-embeddings \
--encoder-normalize-before \
--decoder-normalize-before \
--optimizer adam \
--adam-betas '(0.9, 0.98)' \
--clip-norm 0.5 \
--lr 0.0005 \
--warmup-updates 4000 \
--lr-scheduler inverse_sqrt \
--dropout 0.1 \
--attention-dropout 0.1 \
--num-workers-valid 0 \
--log-interval 10 \
--max-epoch 10 \
--fp16-no-flatten-grads \
--max-tokens 4096 \
--ddp-backend legacy_ddp \
--validate-interval-updates 10000 \
--save-interval-updates 10000 \
--keep-interval-updates 1 \
--no-epoch-checkpoints \
--user-dir ./user_dir \
--best-checkpoint-metric ppl \
--log-format json \
--update-freq 4 \
$save_args \
$model_args \
$moe_args