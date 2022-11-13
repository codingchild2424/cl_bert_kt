#!/bin/bash

cutoff="0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1"

for i in ${cutoff}
do
    python \
    train.py \
    --model_fn cl_monacobert_assist09_spancutoff_${i}.pth \
    --model_name cl_monacobert \
    --dataset_name assist09 \
    --num_encoder 4 \
    --hidden_size 128 \
    --batch_size 128 \
    --grad_acc True \
    --grad_acc_iter 4 \
    --fivefold True \
    --n_epochs 1000 \
    --cl_lambda 0.2 \
    --use_augment True \
    --mask_prob 0 \
    --crop_prob 0 \
    --permute_prob 0 \
    --replace_prob 0 \
    --cutoff_prob ${i}
done
