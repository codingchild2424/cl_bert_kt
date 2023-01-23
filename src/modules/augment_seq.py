import numpy as np
import torch
from random import random, randint, sample
import math

def augment_seq_func(
    q_seqs,
    pid_seqs,
    r_seqs,
    q_diff_seqs,
    pid_diff_seqs,
    mask_seqs,
    num_q,
    num_pid,
    config,
    device
):
    masked_q_seqs = q_seqs
    masked_pid_seqs = pid_seqs
    masked_r_seqs = r_seqs
    masked_q_diff_seqs = q_diff_seqs
    masked_pid_diff_seqs = pid_diff_seqs
    augment_mask_seqs = mask_seqs

    '''
    Concept and Question Mask Function
    - concept과 question에 대한 mask
    '''
    if 1 > config.mask_prob > 0:
        if random() < config.mask_prob:
            masked_q_seqs, masked_pid_seqs, masked_r_seqs, augment_mask_seqs = concept_question_mask_func(
                q_seqs, pid_seqs, r_seqs, mask_seqs, num_q, num_pid, device, config
                )
        else:
            pass
    
    '''
    Crop Fuction
    - 일정 부분만큼 잘라내기
    '''
    if 1 > config.crop_prob > 0:
        if random() < config.crop_prob:
            masked_q_seqs, masked_pid_seqs, masked_r_seqs, augment_mask_seqs = crop_func(
                q_seqs, pid_seqs, r_seqs, mask_seqs, num_q, num_pid, device, config
                )
        else:
            pass

    '''
    Summarization
    - 랜덤으로 일정 부분만 시퀀스를 유지한채로 추출하기
    '''
    if 1 > config.summarize_prob > 0:
        if random() < config.summarize_prob:
            masked_q_seqs, masked_pid_seqs, masked_r_seqs, augment_mask_seqs = summarize_func(
                q_seqs, pid_seqs, r_seqs, mask_seqs, num_q, num_pid, device, config
                )
        else:
            pass

    '''
    Reverse
    - 시퀀스 순서 반대로 섞기
    '''
    if 1 > config.reverse_prob > 0:
        if random() < config.reverse_prob:
            masked_q_seqs, masked_pid_seqs, masked_r_seqs, augment_mask_seqs = reverse_func(
                q_seqs, pid_seqs, r_seqs, mask_seqs, num_q, num_pid, device, config
                )
        else:
            pass

    '''
    Permute Function
    '''
    if 1 > config.permute_prob > 0:
        if random() < config.permute_prob:
            masked_q_seqs, masked_pid_seqs, masked_r_seqs, augment_mask_seqs = permute_func(
                q_seqs, pid_seqs, r_seqs, mask_seqs, num_q, num_pid, device, config
                )
        else:
            pass

    '''
    Segment Permute Function    
    '''
    if 1 > config.segment_permute_prob > 0:
        if random() < config.segment_permute_prob:
            masked_q_seqs, masked_pid_seqs, masked_r_seqs, augment_mask_seqs = segment_permute_func(
                q_seqs, pid_seqs, r_seqs, mask_seqs, num_q, num_pid, device, config
                )
        else:
            pass

    '''
    replace higher diff
    '''
    if 1 > config.replace_higher_diff_prob > 0:
        if random() < config.replace_higher_diff_prob:
            masked_q_seqs, masked_pid_seqs, masked_r_seqs, augment_mask_seqs = replace_higher_diff_func(
                # 난이도 추가
                q_seqs, pid_seqs, r_seqs, q_diff_seqs, mask_seqs, num_q, num_pid, device, config
                )
        else:
            pass

    '''
    replace lower diff
    '''
    if 1 > config.replace_lower_diff_prob > 0:
        if random() < config.replace_lower_diff_prob:
            masked_q_seqs, masked_pid_seqs, masked_r_seqs, augment_mask_seqs = replace_lower_diff_func(
                # 난이도 추가
                q_seqs, pid_seqs, r_seqs, q_diff_seqs, mask_seqs, num_q, num_pid, device, config
                )
        else:
            pass             


    '''
    Concat Two Sequence
    '''
    if 1 > config.concat_seq_prob > 0:
        if random() < config.concat_seq_prob:
            masked_q_seqs, masked_pid_seqs, masked_r_seqs, augment_mask_seqs = concat_seq_func(
                q_seqs, pid_seqs, r_seqs, mask_seqs, num_q, num_pid, device, config
                )
        else:
            pass


    return masked_q_seqs, masked_pid_seqs, masked_r_seqs, masked_q_diff_seqs, masked_pid_diff_seqs, augment_mask_seqs


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
    #if 0 < config.crop_prob < 1:

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
            #start_idx = randint(0, real_q_seq_len - crop_seq_len + 1)
            start_idx = randint(0, real_q_seq_len - crop_seq_len)
                
            masked_q_seq = real_q_seq[start_idx : start_idx + crop_seq_len].to(device)
            masked_pid_seq = real_pid_seq[start_idx : start_idx + crop_seq_len].to(device)
            masked_r_seq = real_r_seq[start_idx : start_idx + crop_seq_len].to(device)

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
    masked_pid_seqs = torch.stack(crop_masked_pid_seqs).to(device)
    masked_r_seqs = torch.stack(crop_masked_r_seqs).to(device)
    augment_mask_seqs = torch.stack(crop_mask_seqs).to(device)

    return masked_q_seqs, masked_pid_seqs, masked_r_seqs, augment_mask_seqs


def summarize_func(q_seqs, pid_seqs, r_seqs, mask_seqs, num_q, num_pid, device, config):

    summarize_masked_q_seqs = []
    summarize_masked_pid_seqs = []
    summarize_masked_r_seqs = []
    summarize_mask_seqs = []

    for q_seq, pid_seq, r_seq, mask_seq in zip(q_seqs, pid_seqs, r_seqs, mask_seqs):
        q_len = q_seq.size(0)
        real_q_seq = torch.masked_select(q_seq.to(device), mask_seq.to(device)).cpu()
        real_q_seq_len = real_q_seq.size(0)

        real_pid_seq = torch.masked_select(pid_seq.to(device), mask_seq.to(device)).cpu()
        real_r_seq = torch.masked_select(r_seq.to(device), mask_seq.to(device)).cpu()

        summarize_seq_len = math.floor(config.summarize_prob * real_q_seq_len)

        # real_q_seq_len <= 5 면 pass
        if real_q_seq_len <= 5:
            summarize_masked_q_seqs.append(q_seq.to(device))
            summarize_masked_pid_seqs.append(pid_seq.to(device))
            summarize_masked_r_seqs.append(r_seq.to(device))
            summarize_mask_seqs.append(mask_seq.to(device))
        # real_q_seq_len > 5        
        else:
            all_indices = range(real_q_seq_len)
            random_indices = torch.tensor(sample(all_indices, summarize_seq_len))
            
            sorted_random_indices = sorted(random_indices)
            #masked_q_seq = [real_q_seq[idx] for idx in sorted_random_indices]
            masked_q_seq = torch.gather(real_q_seq, -1, random_indices).to(device)
            masked_pid_seq = torch.gather(real_pid_seq, -1, random_indices).to(device)
            masked_r_seq = torch.gather(real_r_seq, -1, random_indices).to(device)

            pad_len = q_len - summarize_seq_len

            pad_seq = torch.full((1, pad_len), 0).squeeze(0).to(device)

            pad_q_seq = torch.cat((masked_q_seq, pad_seq), dim=-1)
            pad_pid_seq = torch.cat((masked_pid_seq, pad_seq), dim=-1)
            pad_r_seq = torch.cat((masked_r_seq, pad_seq), dim=-1)
            summarize_mask_seq = torch.cat((
                torch.ones(summarize_seq_len).to(device), 
                pad_seq), dim=-1)
            summarize_mask_seq = torch.tensor(summarize_mask_seq, dtype=torch.bool)

            summarize_masked_q_seqs.append(pad_q_seq.to(device))
            summarize_masked_pid_seqs.append(pad_pid_seq.to(device))
            summarize_masked_r_seqs.append(pad_r_seq.to(device))
            summarize_mask_seqs.append(summarize_mask_seq.to(device))

    masked_q_seqs = torch.stack(summarize_masked_q_seqs).to(device)
    masked_pid_seqs = torch.stack(summarize_masked_pid_seqs).to(device)
    masked_r_seqs = torch.stack(summarize_masked_r_seqs).to(device)
    augment_mask_seqs = torch.stack(summarize_mask_seqs).to(device)

    return masked_q_seqs, masked_pid_seqs, masked_r_seqs, augment_mask_seqs


def reverse_func(q_seqs, pid_seqs, r_seqs, mask_seqs, num_q, num_pid, device, config):
    reverse_masked_q_seqs = []
    reverse_masked_pid_seqs = []
    reverse_masked_r_seqs = []
    reverse_mask_seqs = []

    for q_seq, pid_seq, r_seq, mask_seq in zip(q_seqs, pid_seqs, r_seqs, mask_seqs):
        q_len = q_seq.size(0)
        real_q_seq = torch.masked_select(q_seq.to(device), mask_seq.to(device)).cpu()
        real_pid_seq = torch.masked_select(pid_seq.to(device), mask_seq.to(device)).cpu()
        real_r_seq = torch.masked_select(r_seq.to(device), mask_seq.to(device)).cpu()
        
        real_q_seq_len = real_q_seq.size(0)

        reverse_q_seq = torch.flip(real_q_seq, dims=(-1,)).to(device)
        reverse_pid_seq = torch.flip(real_pid_seq, dims=(-1,)).to(device)
        reverse_r_seq = torch.flip(real_r_seq, dims=(-1,)).to(device)

        pad_len = q_len - real_q_seq_len

        pad_seq = torch.full((1, pad_len), 0).squeeze(0).to(device)

        pad_q_seq = torch.cat((reverse_q_seq, pad_seq), dim=-1)
        pad_pid_seq = torch.cat((reverse_pid_seq, pad_seq), dim=-1)
        pad_r_seq = torch.cat((reverse_r_seq, pad_seq), dim=-1)
        reverse_mask_seq = torch.cat((
            torch.ones(real_q_seq_len).to(device), 
            pad_seq), dim=-1)
        reverse_mask_seq = torch.tensor(reverse_mask_seq, dtype=torch.bool)

        reverse_masked_q_seqs.append(pad_q_seq.to(device))
        reverse_masked_pid_seqs.append(pad_pid_seq.to(device))
        reverse_masked_r_seqs.append(pad_r_seq.to(device))
        reverse_mask_seqs.append(reverse_mask_seq.to(device))

    masked_q_seqs = torch.stack(reverse_masked_q_seqs).to(device)
    masked_pid_seqs = torch.stack(reverse_masked_pid_seqs).to(device)
    masked_r_seqs = torch.stack(reverse_masked_r_seqs).to(device)
    augment_mask_seqs = torch.stack(reverse_mask_seqs).to(device)

    return masked_q_seqs, masked_pid_seqs, masked_r_seqs, augment_mask_seqs


def permute_func(q_seqs, pid_seqs, r_seqs, mask_seqs, num_q, num_pid, device, config):

    permute_masked_q_seqs = []
    permute_masked_pid_seqs = []
    permute_masked_r_seqs = []
    permute_mask_seqs = []

    for q_seq, pid_seq, r_seq, mask_seq in zip(q_seqs, pid_seqs, r_seqs, mask_seqs):
        q_len = q_seq.size(0)

        real_q_seq = torch.masked_select(q_seq.to(device), mask_seq.to(device)).cpu()
        real_pid_seq = torch.masked_select(pid_seq.to(device), mask_seq.to(device)).cpu()
        real_r_seq = torch.masked_select(r_seq.to(device), mask_seq.to(device)).cpu()

        real_q_seq_len = real_q_seq.size(0)

        indices = torch.randperm(real_q_seq_len)

        permute_q_seq = real_q_seq[indices].to(device)
        permute_pid_seq = real_pid_seq[indices].to(device)
        permute_r_seq = real_r_seq[indices].to(device)

        pad_len = q_len - real_q_seq_len

        pad_seq = torch.full((1, pad_len), 0).squeeze(0).to(device)

        pad_q_seq = torch.cat((permute_q_seq, pad_seq), dim=-1)
        pad_pid_seq = torch.cat((permute_pid_seq, pad_seq), dim=-1)
        pad_r_seq = torch.cat((permute_r_seq, pad_seq), dim=-1)

        permute_mask_seq = torch.cat((
            torch.ones(real_q_seq_len).to(device),
            pad_seq), dim=-1)
        permute_mask_seq = torch.tensor(permute_mask_seq, dtype=torch.bool)

        permute_masked_q_seqs.append(pad_q_seq.to(device))
        permute_masked_pid_seqs.append(pad_pid_seq.to(device))
        permute_masked_r_seqs.append(pad_r_seq.to(device))
        permute_mask_seqs.append(permute_mask_seq.to(device))

    masked_q_seqs = torch.stack(permute_masked_q_seqs).to(device)
    masked_pid_seqs = torch.stack(permute_masked_pid_seqs).to(device)
    masked_r_seqs = torch.stack(permute_masked_r_seqs).to(device)
    augment_mask_seqs = torch.stack(permute_mask_seqs).to(device)

    return masked_q_seqs, masked_pid_seqs, masked_r_seqs, augment_mask_seqs


def segment_permute_func(q_seqs, pid_seqs, r_seqs, mask_seqs, num_q, num_pid, device, config):
    
    segment_permute_masked_q_seqs = []
    segment_permute_masked_pid_seqs = []
    segment_permute_masked_r_seqs = []
    segment_permute_mask_seqs = []

    for q_seq, pid_seq, r_seq, mask_seq in zip(q_seqs, pid_seqs, r_seqs, mask_seqs):
        q_len = q_seq.size(0)

        real_q_seq = torch.masked_select(q_seq.to(device), mask_seq.to(device)).cpu()
        real_pid_seq = torch.masked_select(pid_seq.to(device), mask_seq.to(device)).cpu()
        real_r_seq = torch.masked_select(r_seq.to(device), mask_seq.to(device)).cpu()

        real_q_seq_len = real_q_seq.size(0)

        if real_q_seq_len <= 5:
            segment_permute_masked_q_seqs.append(q_seq.to(device))
            segment_permute_masked_pid_seqs.append(pid_seq.to(device))
            segment_permute_masked_r_seqs.append(r_seq.to(device))
            segment_permute_mask_seqs.append(mask_seq.to(device))
        else:
            # make two segment and permute and concat two segment

            segment_len = int(real_q_seq_len / 2)
            segment_num = int(real_q_seq_len / segment_len)

            segment_permute_q_seq = torch.zeros(real_q_seq_len, dtype=torch.long).to(device)
            segment_permute_pid_seq = torch.zeros(real_q_seq_len, dtype=torch.long).to(device)
            segment_permute_r_seq = torch.zeros(real_q_seq_len, dtype=torch.long).to(device)

            for i in range(segment_num):
                # even segment
                if i % 2 == 0:
                    segment_permute_q_seq[i*segment_len:(i+1)*segment_len] = real_q_seq[ (i*segment_len) + segment_len : (i+1)*segment_len + segment_len]
                    segment_permute_pid_seq[i*segment_len:(i+1)*segment_len] = real_pid_seq[ (i*segment_len) + segment_len : (i+1)*segment_len + segment_len]
                    segment_permute_r_seq[i*segment_len:(i+1)*segment_len] = real_r_seq[ (i*segment_len) + segment_len : (i+1)*segment_len + segment_len]
                # odd segment
                else:
                    segment_permute_q_seq[i*segment_len:(i+1)*segment_len] = real_q_seq[ (i*segment_len) - segment_len : (i+1)*segment_len - segment_len]
                    segment_permute_pid_seq[i*segment_len:(i+1)*segment_len] = real_pid_seq[ (i*segment_len) - segment_len : (i+1)*segment_len - segment_len]
                    segment_permute_r_seq[i*segment_len:(i+1)*segment_len] = real_r_seq[ (i*segment_len) - segment_len : (i+1)*segment_len - segment_len]

            pad_len = q_len - real_q_seq_len

            pad_seq = torch.full((1, pad_len), 0).squeeze(0).to(device)
            pad_q_seq = torch.cat((segment_permute_q_seq, pad_seq), dim=-1)
            pad_pid_seq = torch.cat((segment_permute_pid_seq, pad_seq), dim=-1)
            pad_r_seq = torch.cat((segment_permute_r_seq, pad_seq), dim=-1)
            segment_mask_seq = torch.cat((
                torch.ones(real_q_seq_len).to(device),
                pad_seq), dim=-1)
            segment_mask_seq = torch.tensor(segment_mask_seq, dtype=torch.bool)

            segment_permute_masked_q_seqs.append(pad_q_seq.to(device))
            segment_permute_masked_pid_seqs.append(pad_pid_seq.to(device))
            segment_permute_masked_r_seqs.append(pad_r_seq.to(device))
            segment_permute_mask_seqs.append(segment_mask_seq.to(device))

    masked_q_seqs = torch.stack(segment_permute_masked_q_seqs).to(device)
    masked_pid_seqs = torch.stack(segment_permute_masked_pid_seqs).to(device)
    masked_r_seqs = torch.stack(segment_permute_masked_r_seqs).to(device)
    segment_permute_mask_seqs = torch.stack(segment_permute_mask_seqs).to(device)

    return masked_q_seqs, masked_pid_seqs, masked_r_seqs, segment_permute_mask_seqs


def replace_higher_diff_func(q_seqs, pid_seqs, r_seqs, q_diff_seqs, mask_seqs, num_q, num_pid, device, config):
    
    replace_higher_masked_q_seqs = []
    replace_higher_masked_pid_seqs = []
    replace_higher_masked_r_seqs = []
    replace_higher_mask_seqs = []

    for q_seq, pid_seq, r_seq, q_diff_seq, mask_seq in zip(q_seqs, pid_seqs, r_seqs, q_diff_seqs, mask_seqs):

        q_len = q_seq.size(0)

        real_q_seq = torch.masked_select(q_seq.to(device), mask_seq.to(device)).to(device)
        real_pid_seq = torch.masked_select(pid_seq.to(device), mask_seq.to(device)).to(device)
        real_r_seq = torch.masked_select(r_seq.to(device), mask_seq.to(device)).to(device)
        real_q_diff_seq = torch.masked_select(q_diff_seq.to(device), mask_seq.to(device)).to(device)

        real_q_seq_len = real_q_seq.size(0)

        if real_q_seq_len <= 5:
            replace_higher_masked_q_seqs.append(q_seq.to(device))
            replace_higher_masked_pid_seqs.append(pid_seq.to(device))
            replace_higher_masked_r_seqs.append(r_seq.to(device))
            replace_higher_mask_seqs.append(mask_seq.to(device))
        else:
            replace_len = int(real_q_seq_len * config.replace_higher_diff_prob)
            # choose indexes which real_q_diff_seq is lower than others, a number of replace_len
            replace_lower_indexes = torch.topk(real_q_diff_seq, replace_len, largest=False)[1]

            replace_higher_indexes = torch.topk(real_q_diff_seq, replace_len, largest=True)[1]

            # replace the real_q_seq with the indexes of replace_lower_indexes to the real_q_seq with the indexes of replace_higher_indexes
            for i in range(replace_len):
                real_q_seq[replace_lower_indexes[i]] = real_q_seq[replace_higher_indexes[i]]
            
            pad_len = q_len - real_q_seq_len

            pad_seq = torch.full((1, pad_len), 0).squeeze(0).to(device)
            pad_q_seq = torch.cat((real_q_seq, pad_seq), dim=-1)
            pad_pid_seq = torch.cat((real_pid_seq, pad_seq), dim=-1)
            pad_r_seq = torch.cat((real_r_seq, pad_seq), dim=-1)
            replace_mask_seq = torch.cat((
                torch.ones(real_q_seq_len).to(device),
                pad_seq), dim=-1)
            replace_mask_seq = torch.tensor(replace_mask_seq, dtype=torch.bool)

            replace_higher_masked_q_seqs.append(pad_q_seq.to(device))
            replace_higher_masked_pid_seqs.append(pad_pid_seq.to(device))
            replace_higher_masked_r_seqs.append(pad_r_seq.to(device))
            replace_higher_mask_seqs.append(replace_mask_seq.to(device))

    masked_q_seqs = torch.stack(replace_higher_masked_q_seqs).to(device)
    masked_pid_seqs = torch.stack(replace_higher_masked_pid_seqs).to(device)
    masked_r_seqs = torch.stack(replace_higher_masked_r_seqs).to(device)
    replace_higher_mask_seqs = torch.stack(replace_higher_mask_seqs).to(device)

    return masked_q_seqs, masked_pid_seqs, masked_r_seqs, replace_higher_mask_seqs


def replace_lower_diff_func(q_seqs, pid_seqs, r_seqs, q_diff_seqs, mask_seqs, num_q, num_pid, device, config):

    replace_lower_masked_q_seqs = []
    replace_lower_masked_pid_seqs = []
    replace_lower_masked_r_seqs = []
    replace_lower_mask_seqs = []

    for q_seq, pid_seq, r_seq, q_diff_seq, mask_seq in zip(q_seqs, pid_seqs, r_seqs, q_diff_seqs, mask_seqs):
            
            q_len = q_seq.size(0)
    
            real_q_seq = torch.masked_select(q_seq.to(device), mask_seq.to(device)).to(device)
            real_pid_seq = torch.masked_select(pid_seq.to(device), mask_seq.to(device)).to(device)
            real_r_seq = torch.masked_select(r_seq.to(device), mask_seq.to(device)).to(device)
            real_q_diff_seq = torch.masked_select(q_diff_seq.to(device), mask_seq.to(device)).to(device)
    
            real_q_seq_len = real_q_seq.size(0)
    
            if real_q_seq_len <= 5:
                replace_lower_masked_q_seqs.append(q_seq.to(device))
                replace_lower_masked_pid_seqs.append(pid_seq.to(device))
                replace_lower_masked_r_seqs.append(r_seq.to(device))
                replace_lower_mask_seqs.append(mask_seq.to(device))
            else:
                replace_len = int(real_q_seq_len * config.replace_lower_diff_prob)
                # choose indexes which real_q_diff_seq is lower than others, a number of replace_len
                replace_lower_indexes = torch.topk(real_q_diff_seq, replace_len, largest=False)[1]
    
                replace_higher_indexes = torch.topk(real_q_diff_seq, replace_len, largest=True)[1]
    
                # replace the real_q_seq with the indexes of replace_lower_indexes to the real_q_seq with the indexes of replace_higher_indexes
                for i in range(replace_len):
                    real_q_seq[replace_higher_indexes[i]] = real_q_seq[replace_lower_indexes[i]]
                
                pad_len = q_len - real_q_seq_len
    
                pad_seq = torch.full((1, pad_len), 0).squeeze(0).to(device)
                pad_q_seq = torch.cat((real_q_seq, pad_seq), dim=-1)
                pad_pid_seq = torch.cat((real_pid_seq, pad_seq), dim=-1)
                pad_r_seq = torch.cat((real_r_seq, pad_seq), dim=-1)
                replace_mask_seq = torch.cat((
                    torch.ones(real_q_seq_len).to(device),
                    pad_seq), dim=-1)
                replace_mask_seq = torch.tensor(replace_mask_seq, dtype=torch.bool)
    
                replace_lower_masked_q_seqs.append(pad_q_seq.to(device))
                replace_lower_masked_pid_seqs.append(pad_pid_seq.to(device))
                replace_lower_masked_r_seqs.append(pad_r_seq.to(device))
                replace_lower_mask_seqs.append(replace_mask_seq.to(device))

    masked_q_seqs = torch.stack(replace_lower_masked_q_seqs).to(device)
    masked_pid_seqs = torch.stack(replace_lower_masked_pid_seqs).to(device)
    masked_r_seqs = torch.stack(replace_lower_masked_r_seqs).to(device)
    replace_lower_mask_seqs = torch.stack(replace_lower_mask_seqs).to(device)

    return masked_q_seqs, masked_pid_seqs, masked_r_seqs, replace_lower_mask_seqs


def concat_seq_func(q_seqs, pid_seqs, r_seqs, mask_seqs, num_q, num_pid, device, config):

    # choose num is q_seqs.size(0) * config.concat_prob
    concat_num = int(q_seqs.size(0) * config.concat_seq_prob)

    # make real_q_seqs
    real_q_seqs = []
    real_pid_seqs = []
    real_r_seqs = []
    real_mask_seqs = []

    for i in range(q_seqs.size(0)):
        real_q_seq = torch.masked_select(q_seqs[i].to(device), mask_seqs[i].to(device)).to(device)
        real_pid_seq = torch.masked_select(pid_seqs[i].to(device), mask_seqs[i].to(device)).to(device)
        real_r_seq = torch.masked_select(r_seqs[i].to(device), mask_seqs[i].to(device)).to(device)
        real_mask_seq = torch.masked_select(mask_seqs[i].to(device), mask_seqs[i].to(device)).to(device)
        real_q_seqs.append(real_q_seq.to(device))
        real_pid_seqs.append(real_pid_seq.to(device))
        real_r_seqs.append(real_r_seq.to(device))
        real_mask_seqs.append(real_mask_seq.to(device))

    # choose two random q_seq in q_seqs, if sum of q_seq_len is less than config.max_seq_len, concat them
    # but the whole q_seqs.size(0) have to be same at first
    concat_q_seqs = []
    concat_pid_seqs = []
    concat_r_seqs = []
    concat_mask_seqs = []

    for i in range(concat_num):
        q_seq1_index = randint(0, q_seqs.size(0) - 1)
        q_seq2_index = randint(0, q_seqs.size(0) - 1)
        q_seq1_len = real_q_seqs[q_seq1_index].size(0)
        q_seq2_len = real_q_seqs[q_seq2_index].size(0)
        if q_seq1_len + q_seq2_len <= config.max_seq_len:
            concat_q_seq = torch.cat((real_q_seqs[q_seq1_index], real_q_seqs[q_seq2_index]), dim=-1)
            concat_pid_seq = torch.cat((real_pid_seqs[q_seq1_index], real_pid_seqs[q_seq2_index]), dim=-1)
            concat_r_seq = torch.cat((real_r_seqs[q_seq1_index], real_r_seqs[q_seq2_index]), dim=-1)
            concat_mask_seq = torch.cat((real_mask_seqs[q_seq1_index], real_mask_seqs[q_seq2_index]), dim=-1)

            # add pad, if concat_q_seq is less than config.max_seq_len
            if concat_q_seq.size(0) < config.max_seq_len:
                pad_len = config.max_seq_len - concat_q_seq.size(0)
                pad_seq = torch.full((1, pad_len), 0).squeeze(0).to(device)
                concat_q_seq = torch.cat((concat_q_seq, pad_seq), dim=-1)
                concat_pid_seq = torch.cat((concat_pid_seq, pad_seq), dim=-1)
                concat_r_seq = torch.cat((concat_r_seq, pad_seq), dim=-1)
                concat_mask_seq = torch.cat((concat_mask_seq, pad_seq), dim=-1)

            concat_q_seqs.append(concat_q_seq.to(device))
            concat_pid_seqs.append(concat_pid_seq.to(device))
            concat_r_seqs.append(concat_r_seq.to(device))
            concat_mask_seqs.append(concat_mask_seq.to(device))

    # pad the real_q_seqs and rename to pad_q_seqs
    pad_q_seqs = []
    pad_pid_seqs = []
    pad_r_seqs = []
    pad_mask_seqs = []

    for i in range(q_seqs.size(0)):
        if real_q_seqs[i].size(0) < config.max_seq_len:
            pad_len = config.max_seq_len - real_q_seqs[i].size(0)
            pad_seq = torch.full((1, pad_len), 0).squeeze(0).to(device)
            pad_q_seq = torch.cat((real_q_seqs[i], pad_seq), dim=-1)
            pad_pid_seq = torch.cat((real_pid_seqs[i], pad_seq), dim=-1)
            pad_r_seq = torch.cat((real_r_seqs[i], pad_seq), dim=-1)
            pad_mask_seq = torch.cat((real_mask_seqs[i], pad_seq), dim=-1)
        else:
            pad_q_seq = real_q_seqs[i]
            pad_pid_seq = real_pid_seqs[i]
            pad_r_seq = real_r_seqs[i]
            pad_mask_seq = real_mask_seqs[i]

        pad_q_seqs.append(pad_q_seq.to(device))
        pad_pid_seqs.append(pad_pid_seq.to(device))
        pad_r_seqs.append(pad_r_seq.to(device))
        pad_mask_seqs.append(pad_mask_seq.to(device))

    # replace concat_num of real_q_seqs to concat_q_seqs
    for i in range(concat_num):
        try:
            pad_q_seqs[i] = concat_q_seqs[i]
            pad_pid_seqs[i] = concat_pid_seqs[i]
            pad_r_seqs[i] = concat_r_seqs[i]
            pad_mask_seqs[i] = concat_mask_seqs[i]
        except:
            pass

    # convert list to tensor
    pad_q_seqs = torch.stack(pad_q_seqs, dim=0)
    pad_pid_seqs = torch.stack(pad_pid_seqs, dim=0)
    pad_r_seqs = torch.stack(pad_r_seqs, dim=0)
    pad_mask_seqs = torch.stack(pad_mask_seqs, dim=0)

    return pad_q_seqs, pad_pid_seqs, pad_r_seqs, pad_mask_seqs




    
