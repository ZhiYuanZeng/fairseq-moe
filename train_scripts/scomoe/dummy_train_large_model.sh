#!/bin/bash
#SBATCH --ntasks-per-node 8
#SBATCH --cpus-per-task 9
#SBATCH --gpus-per-node 8
#SBATCH -w 16,17,18,19

export NCCL_DEBUG=INFO
export NCCL_BUFFSIZE=1048576

python=/path/to/python
NUM_EXPERTS=$(expr ${SLURM_JOB_NUM_NODES} \* 8)
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

scomoe_feat_args="--arch scomoe \
          --user-dir ./user_dir \
          --scomoe-type feat \
          --ratio1 0.5 \
          --ratio2 0."

scomoe_seq_args="--arch scomoe \
          --user-dir ./user_dir \
          --scomoe-type seq \
          --ratio1 0.5 \
          --ratio2 0. \
          --token-cluster \
          --df-gate-type softmax"

srun $python train.py \
--task dummy_mt \
--fp16 \
--distributed-port 12345 \
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
--max-tokens $bsz \
--ddp-backend legacy_ddp \
--validate-interval-updates 10000 \
--save-interval-updates 10000 \
--keep-interval-updates 1 \
--no-epoch-checkpoints \
--user-dir ./user_dir \
--best-checkpoint-metric ppl \
--log-format json \
--update-freq 4 \
$model_args \
--no-save \
--dataset-size 500000 \
--disable-validation \
--record-a2a-perf-stats \
--max-epoch 1 \
$moe_args 