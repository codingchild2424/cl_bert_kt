#!/bin/bash

# augmentation True

dataset_names="assist09 algebra05 algebra06 ednet"

for dataset_name in $dataset_names
do
    CUDA_VISIBLE_DEVICES=0 python \
    train.py \
    --model_fn cl_monacobert_cl_0_1_no_augment_${dataset_name}_ablation.pth \
    --use_mps_gpu False \
    --model_name cl_monacobert \
    --dataset_name $dataset_name \
    --num_encoder 4 \
    --hidden_size 128 \
    --batch_size 512 \
    --grad_acc True \
    --grad_acc_iter 1 \
    --fivefold True \
    --n_epochs 1000 \
    --cl_lambda 0.1 \
    --use_augment True \
    --loader_type SIM_DIFF_LOADER \
    --mask_prob 0.1 \
    --crop_prob 0.2 \
    --summarize_prob 0.1 \
    --reverse_prob 0 \
    --permute_prob 0 \
    --segment_permute_prob 0 \
    --replace_higher_diff_prob 0 \
    --replace_lower_diff_prob 0 \
    --concat_seq_prob 0 \
    --use_cutoff False \
    --use_span_cutoff False \
    --cutoff_prob 0 \
    --bert_model_name beomi/kobert
done


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