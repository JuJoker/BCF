#! /bin/bash
echo 'CNNLSTM train process...'

seq_len=720
model_name=CNNLSTM

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
  for pred_len in 12 24 36 48 72 84 96 108 120 132 144 156 168
  do
    echo $filename_$pred_len_'train_start'
    python -u /home/yy/BCF/run_longExp.py \
    --custom_model 1 \
    --root_path /home/yy/BCF/dataset/ \
    --data_path $filename \
    --is_training 1 \
    --model_id train \
    --pred_len $pred_len \
    --model $model_name \
    --data building_data >/home/yy/BCF/logs/BuildingEnergyPredict/$model_name'_'$filename'_'$pred_len.log
    echo $filename_$pred_len_'train_end'
  done
done

