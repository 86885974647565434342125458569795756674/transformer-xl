#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

echo 'Run training...'
python train.py \
        --cuda \
        --data ../data/wikitext-103/ \
        --dataset wt103 \
        --adaptive \
        --n_layer 8 \
        --d_model 200 \
        --n_head 5 \
        --d_head 20 \
        --d_inner 1000 \
        --dropout 0.1 \
        --dropatt 0.0 \
        --optim adam \
        --lr 0.00012 \
        --warmup_step 0 \
        --max_step 10000 \
        --tgt_len 75 \
        --mem_len 0 \
        --eval_tgt_len 75 \
        --batch_size 30 \
        --log-interval 100 \
        --eval-interval 100 \
        --max_eval_step 100 \
        --work_dir /share/wt103_small_vanilla \
        --attn_type 2 
