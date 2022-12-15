#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python3.6 main_inpaint.py \
--gpus '0' \
--exp_id "smile_fix" \
--mode sample \
--eval_iter 120000 \
--preload_dataset false --cache_dataset false \
--use_tensorboard false --save_loss true \
--keep_all_models true \
--dataset CelebA \
--batch_size 1 --img_size 128 --eval_batch_size 1 \
--train_path archive/celeba_smile_sample/train \
--eval_path archive/celeba_smile_sample/train \
--test_path archive/celeba_smile_sample/test \
--mask_not_shuffle false \
--model_type 'SAC' \
--cross_attention true \
--mask_type 'FIX' \
--rand_mask_path 'mask/random_masks' \
--fix_mask_path 'mask/fix_mask/half_under_big.png'
