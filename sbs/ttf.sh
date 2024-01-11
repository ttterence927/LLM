#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
seq_len=168
model=PatchTST

for a in 100
do
    for b in 12
    do
        python main.py \
            --root_path ./ \
            --data_path ttf.data \
            --model_id "ETTh2_${model}_\${gpt_layer}_${seq_len}_${b}_${a}" \
            --data ttf \
            --seq_len "$seq_len" \
            --label_len 168 \
            --pred_len "$b" \
            --batch_size 32 \
            --decay_fac 0.5 \
            --learning_rate 0.0005 \
            --train_epochs 10 \
            --d_model 168 \
            --n_heads 4 \
            --d_ff 256 \
            --dropout 0.2 \
            --enc_in 7 \
            --c_out 7 \
            --freq 0 \
            --patch_size 16 \
            --stride 8 \
            --percent "$a" \
            --gpt_layer 6 \
            --itr 1 \
            --model "$model" \
            --cos 1 \
            --tmax 20 \
            --pretrain 1 \
            --is_gpt 1 \
            --target y \
	        --loss_func tildeq \
            --num_classes 2
            -
    done
done