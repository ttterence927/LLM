# Shell script for setting up directories and running experiments

# Create necessary directories
mkdir -p ./logs
mkdir -p ./logs/LongForecasting

# Define base parameters
model_name="GPT4TS"
root_path_name="./dataset/ETT-small/"
data_path_name="ETTh1.csv"
model_id_name="ett_h"
data_name="ett_h"
random_seed=2021

# Loop through different configurations
for dim in 32; do
    for seq_len in 720; do
        for lr in 0.00005; do
            for pred_len in 192 336 720; do
                for scale in 10000; do
                    # Run experiment with current configuration
                    python -u run_longExp.py \
                        --is_training 1 \
			--loss MSE \
                        --root_path "$root_path_name" \
                        --task_name 'long_term_forecast' \
                        --data_path "$data_path_name" \
                        --model_id "${model_id_name}_${seq_len}_${pred_len}" \
                        --model "$model_name" \
                        --data "$data_name" \
			--percent 100 \
                        --features M \
                        --seq_len "$seq_len" \
                        --label_len "$seq_len" \
                        --pred_len "$pred_len" \
                        --enc_in 7 \
                        --e_layers 2 \
                        --n_heads 2 \
                        --d_model 768 \
                        --d_ff 32 \
                        --head_dropout 0 \
                        --adapter_dropout 0.4 \
                        --patch_len 16  \
                        --stride 8 \
                        --des 'Exp' \
                        --train_epochs 100 \
                        --patience 2 \
                        --itr 1 \
                        --batch_size 20 \
                        --learning_rate "$lr" \
                        --warmup_epochs 0 \
                        --scale "$scale" \
                        --gpt_layers 6 \
                        --spect_adapter_layer 6 \
                        --adapter_layer 6 \
                        --T_type 1 \
                        --C_type 1 \
                        --adapter_dim "$dim" \
			--dropout 0.4
                done
            done
        done
    done
done
