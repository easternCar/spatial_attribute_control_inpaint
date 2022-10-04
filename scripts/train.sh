#!/bin/bash
CUDA_VISIBLE_DEVICES=1,2,3 python3.6 main_inpaint.py \
--gpus '0,1,2' \
--exp_id "smile_fixed_id_pixel00_adv1_inpaint5_attr5_cos1" \
--mode train \
--start_iter 0 --end_iter 150000 \
--preload_dataset false --cache_dataset false \
--use_tensorboard false --save_loss true \
--keep_all_models true \
--dataset CelebA \
--batch_size 12 --img_size 128 \
--train_path ../SAC_GAN/archive/celeba_noalign_smiling_all/train \
--eval_path ../SAC_GAN/archive/celeba_noalign_smiling_all/train \
--test_path ../SAC_GAN/archive/celeba_noalign_smiling_all/test \
--model_type 'SAC' \
--mask_type 'FIX' \
--fix_mask_path 'mask/face_mask.png' \
--cross_attention true \
--save_every 5000