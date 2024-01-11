export CUDA_VISIBLE_DEVICES=0

seq_len=720
model=PatchTST

for percent in 100
do
for pred_len in 96
do

python main.py \
    --root_path ./ \
    --data_path ETTh1.csv \
    --model_id ETTh1_$model'_'$gpt_layer'_'$seq_len'_'$pred_len'_'$percent \
    --data ett_h \
    --features M \
    --seq_len $seq_len \
    --label_len 168 \
    --pred_len $pred_len \
    --batch_size 30 \
    --lradj type4 \
    --learning_rate 0.0001 \
    --train_epochs 10 \
    --decay_fac 0.5 \
    --d_model 768 \
    --n_heads 4 \
    --d_ff 32 \
    --dropout 0.3 \
    --enc_in 7 \
    --c_out 7 \
    --freq 0 \
    --patch_size 16 \
    --stride 8 \
    --percent $percent \
    --gpt_layer 6 \
    --itr 1 \
    --model $model \
    --cos 1 \
    --tmax 20 \
    --pretrain 1 \
    --is_gpt 1 \
    --feature_num 1 \
    --target 'OT' \
    --num_classes 4 \
    --loss_func mse \
    --class_ratio 0.05 \
    --patience 4
done
done