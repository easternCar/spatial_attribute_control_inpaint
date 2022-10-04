#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python3.6 main_inpaint.py \
--gpus '0' \
--exp_id "mask_fixed_pixel00_adv1_inpaint5_attr5_cos1" \
--mode sample \
--eval_iter 16000 \
--preload_dataset false --cache_dataset false \
--use_tensorboard false --save_loss true \
--keep_all_models true \
--dataset CelebA \
--batch_size 1 --img_size 128 --eval_batch_size 1 \
--train_path ./archive/celeba_noalign_mask/train \
--eval_path ./archive/celeba_noalign_mask/train \
--test_path ./archive/celeba_noalign_mask/test \
--mask_not_shuffle false \
--model_type 'SAC' \
--cross_attention true \
--mask_type 'FIX' \
--rand_mask_path 'mask_code/deepfill_temp_mask_few' \
--fix_mask_path 'mask_code/fix_mask/half_under_v2.png'
#--fix_mask_path 'mask_code/deepfill_eval_mask/3028.png'
#--fix_mask_path 'mask_code/fix_mask/face_mask_v3.png'
