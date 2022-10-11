import numpy as np
import torch
from random import random, randint


def augment_seq_func(
    q_seqs,
    pid_seqs,
    mask_seqs,
    num_q,
    num_pid,
    config
):
    # config.concept_mask_prob
    # config.crop_prob
    # config.permute_prob
    # config.replace_prob
    # config.max_seq_len

    masked_q_seqs = []
    masked_pid_seqs = []
    
    # concept mask

    concept_mask_token = num_q + 2 # 안전하게 가장 큰 수보다 100 크게 설정
    question_mask_token = num_pid + 2

    if config.mask_prob > 0:
        # concept_mask

        for q_seq, mask_seq in zip(q_seqs, mask_seqs):
            q_len = q_seq.size(0)
            real_q_seq = torch.masked_select(q_seq, mask_seq).cpu()
            real_q_seq_len = real_q_seq.size(0)

            mlm_idx = np.random.choice(real_q_seq_len, int(real_q_seq_len*0.15), replace=False)

            for idx in mlm_idx:
                if random() < 0.8:
                    real_q_seq[idx] = concept_mask_token
                elif random() < 0.5:
                    real_q_seq[idx] = randint(1, concept_mask_token - 1)

            pad_len = q_len - real_q_seq_len
            pad_seq = torch.full((1, pad_len), 3).squeeze(0)
            pad_q_seq = torch.cat((real_q_seq, pad_seq), dim=-1)
            masked_q_seqs.append(pad_q_seq)

        masked_q_seqs = torch.stack(masked_q_seqs)

        # question_mask
        for pid_seq, mask_seq in zip(pid_seqs, mask_seqs):
            pid_len = pid_seq.size(0)
            real_pid_seq = torch.masked_select(pid_seq, mask_seq).cpu()
            real_pid_seq_len = real_pid_seq.size(0)

            mlm_idx = np.random.choice(real_pid_seq_len, int(real_pid_seq_len*0.15), replace=False)

            for idx in mlm_idx:
                if random() < 0.8:
                    real_pid_seq[idx] = question_mask_token
                elif random() < 0.5:
                    real_pid_seq[idx] = randint(1, question_mask_token - 1)

            pad_len = pid_len - real_pid_seq_len
            pad_seq = torch.full((1, pad_len), 3).squeeze(0)
            pad_pid_seq = torch.cat((real_pid_seq, pad_seq), dim=-1)
            masked_pid_seqs.append(pad_pid_seq)

        masked_pid_seqs = torch.stack(masked_pid_seqs)

    else:
        masked_q_seqs = q_seqs[:]
        masked_pid_seqs = pid_seqs[:]


    return masked_q_seqs, masked_pid_seqs

        



    # question mask


    # Interaction crop

    #pass