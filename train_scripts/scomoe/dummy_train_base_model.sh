python=/path/to/python
embed_dim=512
ffn_dim=2048
num_experts=4
moe_args="--moe-gating-use-fp32 \
        --moe-second-expert-policy all \
        --moe-normalize-expert-grad sqrt_world_size \
        --criterion moe_cross_entropy \
        --moe-gate-loss-wt 0.01 \
        --moe-gate-loss-combine-method sum \
        --moe-batch-prioritized-routing \
        --use-moe-pad-mask \
        --moe-expert-count ${num_experts} \
        --fp16-no-flatten-grads \
        --moe-freq 2 \
        --encoder-embed-dim $embed_dim \
        --decoder-embed-dim $embed_dim \
        --encoder-ffn-embed-dim $ffn_dim \
        --decoder-ffn-embed-dim $ffn_dim \
        --capacity-factor 1.0"

scomoe_seq_args="--arch scomoe \
            --scomoe-type seq \
            --ratio1 0.3 \
            --ratio2 0.7 \
            --token-cluster \
            --temperature 0.5"

scomoe_feat_args="--arch scomoe \
                --scomoe-type feat \
                --token-cluster \
                --ratio1 0.5 \
                --ratio2 0.5"

python train.py \
    --task dummy_mt \
    --arch transformer --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 1.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion cross_entropy \
    --max-tokens 4096 \
    --ddp-backend legacy_ddp \
    --fp16 \
    --encoder-normalize-before \
    --decoder-normalize-before \
    --no-epoch-checkpoints \
    --user-dir scomoe \
    --max-epoch 1 \
    --no-save \
    --disable-validation \
    --log-interval 10 \
    --log-format json \
    --record-a2a-perf-stats \
    --dataset-size 500000 \
    $moe_args \
    $scomoe_feat_args \
    --no-save