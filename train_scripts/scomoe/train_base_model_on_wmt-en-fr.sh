data=/path/to/data_bin
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
        --moe-expert-count 8 \
        --fp16-no-flatten-grads \
        --moe-freq 2"

scomoe_seq_args="--arch scomoe \
        --scomoe-type seq \
        --ratio1 0.5 \
        --ratio2 0.5 \
        --token-cluster \
        --temperature 0.5"

scomoe_feat_args="--arch scomoe \
        --scomoe-type feat \
        --ratio1 0.5 \
        --ratio2 0.5"

python train.py \
    $data \
    --arch transformer --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 1.0 \
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
    --no-epoch-checkpoints \
    --user-dir user_dir \
    $moe_args \
    --eval-bleu \
    --best-checkpoint-metric bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-print-samples \
    --maximize-best-checkpoint-metric \
    --capacity-factor 1.0 \
    --eval-bleu-remove-bpe \
    --eval-bleu-detok moses \
    --max-epoch 10 \
    --valid-subset valid