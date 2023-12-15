#!/bin/bash

echo 'Run training...'
python train.py \
        --cuda \
        --data ../data/enwik8/ \
        --dataset enwik8 \
        --n_layer 6 \
        --d_model 256 \
        --n_head 4 \
        --d_head 32 \
        --d_inner 1024 \
        --dropout 0.1 \
        --dropatt 0.0 \
        --optim adam \
        --lr 0.00012 \
        --warmup_step 0 \
        --max_step 1000 \
        --tgt_len 256 \
        --mem_len 256 \
        --eval_tgt_len 64 \
        --batch_size 11 \
	--eval-interval 100 \
	--log-interval 100 \
	--max_eval_step 500 \
	--work_dir /share/enwik8_small_cyy
