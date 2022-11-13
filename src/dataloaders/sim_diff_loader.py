import numpy as np
import pandas as pd

import torch

from torch.nn.utils.rnn import pad_sequence

from torch.utils.data import Dataset
from utils import pid_collate_fn

from modules.augment_seq import augment_seq_func

class SIM_DIFF_LOADER(Dataset):
    def __init__(self, max_seq_len, dataset_dir, config, idx=None) -> None:
        super(SIM_DIFF_LOADER, self).__init__()

        self.idx = idx

        self.dataset_dir = dataset_dir

        self.config = config
        
        self.q_seqs, self.r_seqs, self.q_list, self.u_list, self.r_list, self.q2idx, \
            self.u2idx, self.pid_seqs, self.pid_list, self.negative_r_seqs, \
                self.diff_seqs, self.diff_list = self.preprocess()

        self.num_u = self.u_list.shape[0]
        self.num_q = self.q_list.shape[0]
        self.num_r = self.r_list.shape[0]
        self.num_pid = self.pid_list.shape[0]
        self.num_diff = self.diff_list.shape[0]

        self.collate_fn = pid_collate_fn

        self.q_seqs, self.r_seqs, self.pid_seqs, self.negative_r_seqs, self.diff_seqs, self.mask_seqs = \
            self.match_seq_len(self.q_seqs, self.r_seqs, self.pid_seqs, self.negative_r_seqs, self.diff_seqs, max_seq_len)

        self.len = len(self.q_seqs)


    def __len__(self):
        return self.len

    def preprocess(self):
        df = pd.read_csv(self.dataset_dir, encoding="ISO-8859-1", sep='\t')
        df = df[(df["correct"] == 0) | (df["correct"] == 1)]

        # zero for padding
        df["user_id"] += 1
        df["skill_id"] += 1
        df["item_id"] += 1

        u_list = np.unique(df["user_id"].values)
        q_list = np.unique(df["skill_id"].values)
        r_list = np.unique(df["correct"].values)
        pid_list = np.unique(df["item_id"].values)

        u2idx = {u: idx for idx, u in enumerate(u_list)}
        q2idx = {q: idx for idx, q in enumerate(q_list)}
        pid2idx = {pid: idx for idx, pid in enumerate(pid_list)} 

        u_idx = np.arange(int(len(u_list)))

        # idx에 맞게 조정
        first_chunk = u_idx[ : int(len(u_list) * 0.2) ]
        second_chunk = u_idx[ int(len(u_list) * 0.2) : int(len(u_list) * 0.4) ]
        third_chunk = u_idx[ int(len(u_list) * 0.4) : int(len(u_list) * 0.6) ]
        fourth_chunk = u_idx[ int(len(u_list) * 0.6) : int(len(u_list) * 0.8) ]
        fifth_chunk = u_idx[ int(len(u_list) * 0.8) : ]

        if self.idx == 0:
            train_u_idx = np.concatenate( (second_chunk, third_chunk, fourth_chunk, fifth_chunk), axis = 0 )
            real_train_u_idx = train_u_idx[ : int( len(train_u_idx) * (1 - self.config.valid_ratio) ) ]
            valid_u_idx = train_u_idx[ int( len(train_u_idx) * (1 - self.config.valid_ratio) ) : ]
            test_u_idx = first_chunk
        elif self.idx == 1:
            train_u_idx = np.concatenate( (first_chunk, third_chunk, fourth_chunk, fifth_chunk), axis = 0 )
            real_train_u_idx = train_u_idx[ : int( len(train_u_idx) * (1 - self.config.valid_ratio) ) ]
            valid_u_idx = train_u_idx[ int( len(train_u_idx) * (1 - self.config.valid_ratio) ) : ]
            test_u_idx = second_chunk
        elif self.idx == 2:
            train_u_idx = np.concatenate( (first_chunk, second_chunk, fourth_chunk, fifth_chunk), axis = 0 )
            real_train_u_idx = train_u_idx[ : int( len(train_u_idx) * (1 - self.config.valid_ratio) ) ]
            valid_u_idx = train_u_idx[ int( len(train_u_idx) * (1 - self.config.valid_ratio) ) : ]
            test_u_idx = third_chunk
        elif self.idx == 3:
            train_u_idx = np.concatenate( (first_chunk, second_chunk, third_chunk, fifth_chunk), axis = 0 )
            real_train_u_idx = train_u_idx[ : int( len(train_u_idx) * (1 - self.config.valid_ratio) ) ]
            valid_u_idx = train_u_idx[ int( len(train_u_idx) * (1 - self.config.valid_ratio) ) : ]
            test_u_idx = fourth_chunk
        elif self.idx == 4:
            train_u_idx = np.concatenate( (first_chunk, second_chunk, third_chunk, fourth_chunk), axis = 0 )
            real_train_u_idx = train_u_idx[ : int( len(train_u_idx) * (1 - self.config.valid_ratio) ) ]
            valid_u_idx = train_u_idx[ int( len(train_u_idx) * (1 - self.config.valid_ratio) ) : ]
            test_u_idx = fifth_chunk

        q_seqs = []
        r_seqs = []
        pid_seqs = []
        negative_r_seqs = [] # 추가

        # for diff
        train_pid_seqs = []
        train_r_seqs = []

        for idx, u in enumerate(u_list):
            df_u = df[df["user_id"] == u]

            q_seq = np.array([q2idx[q] for q in df_u["skill_id"].values])
            r_seq = df_u["correct"].values
            pid_seq = np.array([pid2idx[pid] for pid in df_u["item_id"].values])

            q_seqs.append(q_seq)
            r_seqs.append(r_seq)
            pid_seqs.append(pid_seq)
            negative_r_seqs.append(1 - r_seq)

            if idx in real_train_u_idx:
                train_pid_seqs.extend(pid_seq)
                train_r_seqs.extend(r_seq)

                # train_df
        train_df = pd.DataFrame(
            zip(train_pid_seqs, train_r_seqs), 
            columns = ["pid", "r"]
            )
        # pid_diff
        train_pid_diff = np.round(train_df.groupby('pid')['r'].mean() * 100)
        diff_list = np.unique(train_df.groupby('pid')['r'].mean()) 

        diff_seqs = []

        train_pid_list = np.unique(train_pid_seqs)

        for pid_seq in pid_seqs:

            pid_diff_seq = []

            for pid in pid_seq:
                if pid not in train_pid_list:
                    pid_diff_seq.append(float(75)) # <PAD>
                else:
                    pid_diff_seq.append(train_pid_diff[pid])

            diff_seqs.append(pid_diff_seq)   
        

        return q_seqs, r_seqs, q_list, u_list, r_list, q2idx, u2idx, pid_seqs, pid_list, negative_r_seqs, diff_seqs, diff_list

    def match_seq_len(self, q_seqs, r_seqs, pid_seqs, negative_r_seqs, diff_seqs, max_seq_len, pad_val=-1):

        proc_q_seqs = []
        proc_r_seqs = []
        proc_pid_seqs = []
        proc_negative_r_seqs = []
        proc_diff_seqs = []

        for q_seq, r_seq, pid_seq, negative_r_seq, diff_seq in zip(q_seqs, r_seqs, pid_seqs, negative_r_seqs, diff_seqs):

            i = 0
            while i + max_seq_len < len(q_seq):
                proc_q_seqs.append(q_seq[i:i + max_seq_len])
                proc_r_seqs.append(r_seq[i:i + max_seq_len])
                proc_pid_seqs.append(pid_seq[i:i + max_seq_len])
                proc_negative_r_seqs.append(negative_r_seq[i:i + max_seq_len])
                proc_diff_seqs.append(diff_seq[i:i + max_seq_len])

                i += max_seq_len

            proc_q_seqs.append(
                np.concatenate(
                    [
                        q_seq[i:],
                        np.array([pad_val] * (i + max_seq_len - len(q_seq)))
                    ]
                )
            )
            proc_r_seqs.append(
                np.concatenate(
                    [
                        r_seq[i:],
                        np.array([pad_val] * (i + max_seq_len - len(q_seq)))
                    ]
                )
            )
            proc_pid_seqs.append(
                np.concatenate(
                    [
                        pid_seq[i:],
                        np.array([pad_val] * (i + max_seq_len - len(q_seq)))
                    ]
                )
            )
            proc_negative_r_seqs.append(
                np.concatenate(
                    [
                        negative_r_seq[i:],
                        np.array([pad_val] * (i + max_seq_len - len(q_seq)))
                    ]
                )
            )
            proc_diff_seqs.append(
                np.concatenate(
                    [
                        diff_seq[i:],
                        np.array([pad_val] * (i + max_seq_len - len(q_seq)))
                    ]
                )
            )

        # mask 추출 및 padding 작업
        pad_proc_q_seqs = []
        pad_proc_r_seqs = []
        pad_proc_pid_seqs = []
        pad_proc_negative_r_seqs = []
        pad_proc_diff_seqs = []

        for proc_q_seq, proc_r_seq, proc_pid_seq, proc_negative_r_seq, proc_diff_seq in zip(proc_q_seqs, proc_r_seqs, proc_pid_seqs, proc_negative_r_seqs, proc_diff_seqs):

            #torch.tensor(aug_q_seq_1, dtype=torch.long)
            pad_proc_q_seqs.append(torch.tensor(proc_q_seq, dtype=torch.long))
            pad_proc_r_seqs.append(torch.tensor(proc_r_seq, dtype=torch.long))
            pad_proc_pid_seqs.append(torch.tensor(proc_pid_seq, dtype=torch.long))
            pad_proc_negative_r_seqs.append(torch.tensor(proc_negative_r_seq, dtype=torch.long))
            pad_proc_diff_seqs.append(torch.tensor(proc_diff_seq, dtype=torch.long))

        pad_proc_q_seqs = pad_sequence(
            pad_proc_q_seqs, batch_first=True, padding_value=pad_val
        )
        pad_proc_r_seqs = pad_sequence(
            pad_proc_r_seqs, batch_first=True, padding_value=pad_val
        )
        pad_proc_pid_seqs = pad_sequence(
            pad_proc_pid_seqs, batch_first=True, padding_value=pad_val
        )
        pad_proc_negative_r_seqs = pad_sequence(
            pad_proc_negative_r_seqs, batch_first=True, padding_value=pad_val
        )
        pad_proc_diff_seqs = pad_sequence(
            pad_proc_diff_seqs, batch_first=True, padding_value=pad_val
        )

        mask_seqs = (pad_proc_q_seqs != pad_val)

        pad_proc_q_seqs, pad_proc_r_seqs, \
            pad_proc_pid_seqs, pad_proc_negative_r_seqs, pad_proc_diff_seqs = \
            pad_proc_q_seqs * mask_seqs, pad_proc_r_seqs * mask_seqs, \
                pad_proc_pid_seqs * mask_seqs, pad_proc_negative_r_seqs * mask_seqs, pad_proc_diff_seqs * mask_seqs

        return pad_proc_q_seqs, pad_proc_r_seqs, pad_proc_pid_seqs, pad_proc_negative_r_seqs, pad_proc_diff_seqs, mask_seqs


    # # augmentation을 위한 추가
    # def __getitem_internal__(self, index):

    #     return {
    #         "concepts": self.q_seqs[index], 
    #         "responses": self.r_seqs[index], 
    #         "questions": self.pid_seqs[index], 
    #         "negative_responses": self.negative_r_seqs[index], 
    #         "masks": self.mask_seqs[index]
    #         }

    def __getitem__(self, index):
        #return self.__getitem_internal__(index)
        return {
            "concepts": self.q_seqs[index], 
            "responses": self.r_seqs[index], 
            "questions": self.pid_seqs[index], 
            "negative_responses": self.negative_r_seqs[index],
            "difficult": self.diff_seqs[index],
            "masks": self.mask_seqs[index]
            }