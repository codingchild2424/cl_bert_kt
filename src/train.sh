#!/bin/bash

python \
train.py \
--model_fn cl_monacobert_assist09.pth \
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
--use_augment True
