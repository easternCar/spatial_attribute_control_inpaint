#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python3.6 main_inpaint.py \
--gpus '0' \
--exp_id "smile_fix" \
--mode train \
--start_iter 0 --end_iter 150000 \
--preload_dataset false --cache_dataset false \
--use_tensorboard false --save_loss true \
--keep_all_models true \
--dataset CelebA \
--batch_size 12 --img_size 128 \
--train_path archive/celeba_smile_sample/train \
--eval_path archive/celeba_smile_sample/train \
--test_path archive/celeba_smile_sample/test \
--model_type 'SAC' \
--mask_type 'FIX' \
--cross_attention true \
--save_every 5000