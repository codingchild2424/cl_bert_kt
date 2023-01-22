import argparse
import torch

def define_argparser():
    p = argparse.ArgumentParser()

    # model_file_name
    p.add_argument('--model_fn', required=True)

    # basic arguments
    p.add_argument('--gpu_id', type=int, default=0 if torch.cuda.is_available() else -1)
    p.add_argument('--train_ratio', type=float, default=.8)
    p.add_argument('--valid_ratio', type=float, default=.1)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--n_epochs', type=int, default=100)
    p.add_argument('--verbose', type=int, default=2)
    p.add_argument('--learning_rate', type=int, default = 0.001)

    # model, opt, dataset, crit arguments
    p.add_argument('--model_name', type=str, default='cl_monacobert')
    p.add_argument('--optimizer', type=str, default='adam')
    p.add_argument('--dataset_name', type=str, default = 'assist2009_pid')
    p.add_argument('--crit', type=str, default = 'binary_cross_entropy')

    # bidkt's arguments
    p.add_argument('--max_seq_len', type=int, default=100)
    p.add_argument('--num_encoder', type=int, default=12)
    p.add_argument('--hidden_size', type=int, default=128) # 128 is okay
    p.add_argument('--num_head', type=int, default=16) # it will be divided 2(default) in attention class, so actual head num is 8(default)
    p.add_argument('--output_size', type=int, default=1) # KT is binary classification
    p.add_argument('--dropout_p', type=int, default=.1)
    p.add_argument('--use_leakyrelu', type=bool, default=True)

    # cl
    p.add_argument('--cl_lambda', type=float, default=0.2)
    
    # grad_accumulation
    p.add_argument('--grad_acc', type=bool, default=False)
    p.add_argument('--grad_acc_iter', type=int, default=2)

    # five_fold cross validation
    p.add_argument('--fivefold', type=bool, default=False)

    # augmentation
    p.add_argument('--use_augment', type=bool, default=False)
    p.add_argument('--seed', type=float, default=12405)
    # augmentation mask
    p.add_argument('--mask_prob', type=float, default=0.2)
    p.add_argument('--crop_prob', type=float, default=0.3)
    p.add_argument('--summarize_prob', type=float, default=0.3)
    p.add_argument('--reverse_prob', type=float, default=0.3)
    p.add_argument('--permute_prob', type=float, default=0.3)
    p.add_argument('--segment_permute_prob', type=float, default=0.3)

    p.add_argument('--replace_prob', type=float, default=0.3)
    # cutoff
    p.add_argument('--use_cutoff', type=bool, default=False)
    p.add_argument('--use_span_cutoff', type=bool, default=True)
    p.add_argument('--cutoff_prob', type=float, default=0.01)

    

    config = p.parse_args()

    return config