#!/bin/bash

# augmentation True

python \
train.py \
--model_fn cl_monacobert_cl_0_1_no_augment_use_llm.pth \
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
--use_llm_loader True \
--mask_prob 0 \
--crop_prob 0 \
--summarize_prob 0 \
--reverse_prob 0 \
--permute_prob 0 \
--segment_permute_prob 0 \
--replace_higher_diff_prob 0 \
--replace_lower_diff_prob 0 \
--concat_seq_prob 0 \
--use_cutoff False \
--use_span_cutoff False \
--cutoff_prob 0


# probs="0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9"

# for prob in $probs
# do
#     python \
#     train.py \
#     --model_fn cl_monacobert_mask_$prob.pth \
#     --model_name cl_monacobert \
#     --dataset_name assist09 \
#     --num_encoder 4 \
#     --hidden_size 128 \
#     --batch_size 128 \
#     --grad_acc True \
#     --grad_acc_iter 4 \
#     --fivefold True \
#     --n_epochs 1000 \
#     --cl_lambda 0.1 \
#     --use_augment True \
#     --mask_prob $prob \
#     --crop_prob 0 \
#     --summarize_prob 0 \
#     --reverse_prob 0 \
#     --permute_prob 0 \
#     --segment_permute_prob 0 \
#     --replace_higher_diff_prob 0 \
#     --replace_lower_diff_prob 0 \
#     --concat_seq_prob 0 \
#     --use_cutoff False \
#     --use_span_cutoff False \
#     --cutoff_prob 0
# done

