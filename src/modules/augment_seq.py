import numpy as np
import torch
from random import random, randint
import math

def augment_seq_func(
    q_seqs,
    pid_seqs,
    r_seqs,
    diff_seqs,
    mask_seqs,
    num_q,
    num_pid,
    config,
    device
):
    # config.concept_mask_prob
    # config.crop_prob
    # config.permute_prob
    # config.replace_prob
    # config.max_seq_len

    masked_q_seqs = q_seqs
    masked_pid_seqs = pid_seqs
    masked_r_seqs = r_seqs
    masked_diff_seqs = diff_seqs
    augment_mask_seqs = mask_seqs

    if config.mask_prob > 0:
        masked_q_seqs, masked_pid_seqs, masked_r_seqs, augment_mask_seqs = concept_question_mask_func(
            q_seqs, pid_seqs, r_seqs, mask_seqs, num_q, num_pid, device, config
            )

    return masked_q_seqs, masked_pid_seqs, masked_r_seqs, masked_diff_seqs, augment_mask_seqs


def concept_question_mask_func(q_seqs, pid_seqs, r_seqs, mask_seqs, num_q, num_pid, device, config):

    masked_q_seqs = []
    masked_pid_seqs = []
    
    # concept mask

    concept_mask_token = num_q + 2 # 안전하게 가장 큰 수보다 100 크게 설정
    question_mask_token = num_pid + 2

    for q_seq, mask_seq in zip(q_seqs, mask_seqs):
        q_len = q_seq.size(0)
        real_q_seq = torch.masked_select(q_seq, mask_seq).cpu()
        real_q_seq_len = real_q_seq.size(0)

        mlm_idx = np.random.choice(real_q_seq_len, int(real_q_seq_len*0.15), replace=False)

        for idx in mlm_idx:
            if random() < 0.8:
                real_q_seq[idx] = concept_mask_token
            elif random() < 0.9:
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
            elif random() < 0.9:
                real_pid_seq[idx] = randint(1, question_mask_token - 1)

        pad_len = pid_len - real_pid_seq_len
        pad_seq = torch.full((1, pad_len), 3).squeeze(0)
        pad_pid_seq = torch.cat((real_pid_seq, pad_seq), dim=-1)
        masked_pid_seqs.append(pad_pid_seq)

    masked_pid_seqs = torch.stack(masked_pid_seqs)

    masked_r_seqs = r_seqs[:]
    augment_mask_seqs = mask_seqs[:]

    return masked_q_seqs, masked_pid_seqs, masked_r_seqs, augment_mask_seqs


def crop_func(q_seqs, pid_seqs, r_seqs, mask_seqs, num_q, num_pid, device, config):
    # crop
    if 0 < config.crop_prob < 1:

        crop_masked_q_seqs = []
        crop_masked_pid_seqs = []
        crop_masked_r_seqs = []
        crop_mask_seqs = []

        for q_seq, pid_seq, r_seq, mask_seq in zip(q_seqs, pid_seqs, r_seqs, mask_seqs):
            q_len = q_seq.size(0)
            real_q_seq = torch.masked_select(q_seq.to(device), mask_seq.to(device)).cpu()
            real_q_seq_len = real_q_seq.size(0)

            real_pid_seq = torch.masked_select(pid_seq.to(device), mask_seq.to(device)).cpu()
            real_r_seq = torch.masked_select(r_seq.to(device), mask_seq.to(device)).cpu()
            
            crop_seq_len = math.floor(config.crop_prob * real_q_seq_len)

            # real_q_seq_len <= 5 면 pass
            if real_q_seq_len <= 5:
                crop_masked_q_seqs.append(q_seq.to(device))
                crop_masked_pid_seqs.append(pid_seq.to(device))
                crop_masked_r_seqs.append(r_seq.to(device))
                crop_mask_seqs.append(mask_seq.to(device))
            # real_q_seq_len > 5
            else:
                start_idx = randint(0, real_q_seq_len - crop_seq_len + 1)
                
                masked_q_seq = real_q_seq[start_idx : start_idx + crop_seq_len].to(device)
                masked_pid_seq = pid_seq[start_idx : start_idx + crop_seq_len].to(device)
                masked_r_seq = r_seq[start_idx : start_idx + crop_seq_len].to(device)

                pad_len = q_len - crop_seq_len # 잘라낸만큼 패드 추가

                pad_seq = torch.full((1, pad_len), 0).squeeze(0).to(device)

                pad_q_seq = torch.cat((masked_q_seq, pad_seq), dim=-1)
                pad_pid_seq = torch.cat((masked_pid_seq, pad_seq), dim=-1)
                pad_r_seq = torch.cat((masked_r_seq, pad_seq), dim=-1)
                crop_mask_seq = torch.cat((
                    torch.ones(crop_seq_len).to(device), 
                    pad_seq), dim=-1)
                crop_mask_seq = torch.tensor(crop_mask_seq, dtype=torch.bool)

                crop_masked_q_seqs.append(pad_q_seq.to(device))
                crop_masked_pid_seqs.append(pad_pid_seq.to(device))
                crop_masked_r_seqs.append(pad_r_seq.to(device))
                crop_mask_seqs.append(crop_mask_seq.to(device))

        masked_q_seqs = torch.stack(crop_masked_q_seqs).to(device)
        masked_pid_seq = torch.stack(crop_masked_pid_seqs).to(device)
        masked_r_seqs = torch.stack(crop_masked_r_seqs).to(device)
        augment_mask_seqs = torch.stack(crop_mask_seqs).to(device)

            

            # pad 추가하기!!!!
            # # cover the PAD(-1)
            # pad_len = q_len - real_q_seq_len
            # # <PAD> is 3
            # pad_seq = torch.full((1, pad_len), 0).squeeze(0) 
            # # combine the <PAD>
            # pad_q_seq = torch.cat((real_r_seq, pad_seq), dim=-1)

            # # pad 추가하기
            # mask_q_seqs = 
    pass
