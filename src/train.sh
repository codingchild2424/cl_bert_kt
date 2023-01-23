#!/bin/bash

python \
train.py \
--model_fn test.pth \
--model_name cl_monacobert \
--dataset_name assist09 \
--num_encoder 4 \
--hidden_size 128 \
--batch_size 128 \
--grad_acc True \
--grad_acc_iter 4 \
--fivefold True \
--n_epochs 1000 \
--cl_lambda 0.1 \
--use_augment True \
--mask_prob 0 \
--crop_prob 0 \
--summarize_prob 0 \
--reverse_prob 0 \
--permute_prob 0 \
--segment_permute_prob 0 \
--replace_higher_diff_prob 0 \
--replace_lower_diff_prob 0.3 \
--use_cutoff False
