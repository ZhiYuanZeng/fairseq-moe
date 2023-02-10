#!/bin/bash
#SBATCH --nodes 4
#SBATCH --ntasks-per-node 8
#SBATCH --cpus-per-task 9
#SBATCH --gpus-per-node 8
#SBATCH -w 16,17,18,19

export NCCL_DEBUG=INFO
export NCCL_BUFFSIZE=1048576

python=/path/to/python
data=/path/to/data-bin
save_dir=/path/to/save_dir
tensorboard_dir=/path/to/tensorboard_dir

moe_args="--moe-gating-use-fp32 \
        --moe-second-expert-policy all \
        --moe-normalize-expert-grad sqrt_world_size \
        --criterion moe_cross_entropy \
        --moe-gate-loss-wt 0.01 \
        --moe-gate-loss-combine-method sum \
        --moe-batch-prioritized-routing \
        --use-moe-pad-mask \
        --moe-expert-count 32 \
        --fp16-no-flatten-grads \
        --moe-freq 2 \
        --capacity-factor 1.25"

model_args="--arch transformer \
            --encoder-layers 12 \
            --decoder-layers 12 \
            --encoder-embed-dim 1024 \
            --decoder-embed-dim 1024 \
            --encoder-ffn-embed-dim 4096 \
            --decoder-ffn-embed-dim 4096"

scomoe_seq_args="--arch scomoe \
        --scomoe-type seq \
        --ratio1 0.5 \
        --ratio2 0. \
        --token-cluster \
        --temperature 0.5"

scomoe_feat_args="--arch scomoe \
        --scomoe-type feat \
        --ratio1 0.5 \
        --ratio2 0."

srun $python train.py \
    --distributed-port 12345 \
    $data \
    --arch transformer --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.5 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion cross_entropy \
    --max-tokens 4096 \
    --save-dir $save_dir \
    --tensorboard-logdir $tensorboard_dir \
    --ddp-backend legacy_ddp \
    --fp16 \
    --encoder-normalize-before \
    --decoder-normalize-before \
    --user-dir user_dir \
    $moe_args \
    --eval-bleu \
    --best-checkpoint-metric bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-print-samples \
    --maximize-best-checkpoint-metric \
    --eval-bleu-remove-bpe \
    --eval-bleu-detok moses \
    --max-epoch 10 \
    --update-freq 4 \
    $model_args