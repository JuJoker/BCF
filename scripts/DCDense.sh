#! /bin/bash
echo 'DCLinear train process...'

seq_len=720
model_name=DCDense

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
  for pred_len in 6 12 24 48 72 96 120 144 168 336 504 720
  do
    python -u /home/yy/BCF/run_longExp.py \
    --custom_model 1 \
    --root_path /home/yy/BCF/dataset/ \
    --data_path $filename \
    --is_training 1 \
    --model_id train \
    --pred_len $pred_len \
    --model $model_name \
    --checkpoints /home/yy/BCF/checkpoints/ \
    --res_csv_path /home/yy/BCF/ \
    --data building_data >/home/yy/BCF/logs/BuildingEnergyPredict/$model_name'_'$filename'_'$pred_len.log
  done
done

