#! /bin/bash
echo 'Formers train process...'

seq_len=720

if [ ! -d "/home/yy/BCF/logs" ]; then
    mkdir /home/yy/BCF/logs
fi

if [ ! -d "/home/yy/BCF/logs/BuildingEnergyPredict" ]; then
    mkdir /home/yy/BCF/logs/BuildingEnergyPredict
fi

# data file path
directory_path="/home/yy/BCF/dataset"

# get files
for file in "$directory_path"/*
do
  filename=$(basename "$file")
  for model_name in Autoformer Informer Transformer
  do
    for pred_len in 12 24 36 48 72 84 96 108 120 132 144 156 168
    do
      python -u /home/yy/BCF/run_longExp.py \
      --custom_model 0 \
      --is_training 1 \
      --model_id train \
      --root_path /home/yy/BCF/dataset/ \
      --data_path $filename \
      --model $model_name \
      --data custom \
      --seq_len $seq_len \
      --label_len 24 \
      --pred_len $pred_len \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 >/home/yy/BCF/logs/BuildingEnergyPredict/$model_name'_'$filename'_'$pred_len.log
    done
  done
done