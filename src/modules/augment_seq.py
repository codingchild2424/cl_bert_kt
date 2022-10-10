
def augment_seq_func(
    q_seqs,
    r_seqs,
    pid_seqs,
    num_q,
    num_pid,
    config
):
    # config.concept_mask_prob
    # config.crop_prob
    # config.permute_prob
    # config.replace_prob
    # config.max_seq_len
    # config.seed

    masked_q_seqs = []
    masked_r_seqs = []
    masked_pid_seqs = []
    
    if config.concept_mask_prob > 0:
        pass

    pass