import numpy as np
import pandas as pd

import torch

from torch.nn.utils.rnn import pad_sequence

from torch.utils.data import Dataset
from utils import pid_collate_fn

from modules.augment_seq import augment_seq_func

class SIM_LOADER(Dataset):
    def __init__(self, max_seq_len, dataset_dir, config) -> None:
        super(SIM_LOADER, self).__init__()

        self.dataset_dir = dataset_dir

        self.config = config
        
        self.q_seqs, self.r_seqs, self.q_list, self.u_list, self.r_list, self.q2idx, \
            self.u2idx, self.pid_seqs, self.pid_list, self.negative_r_seqs = self.preprocess()

        self.num_u = self.u_list.shape[0]
        self.num_q = self.q_list.shape[0]
        self.num_r = self.r_list.shape[0]
        self.num_pid = self.pid_list.shape[0]
        self.num_diff = None

        self.collate_fn = pid_collate_fn

        self.q_seqs, self.r_seqs, self.pid_seqs, self.negative_r_seqs, self.mask_seqs = \
            self.match_seq_len(self.q_seqs, self.r_seqs, self.pid_seqs, self.negative_r_seqs, max_seq_len)

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

        q_seqs = []
        r_seqs = []
        pid_seqs = []
        negative_r_seqs = [] # 추가

        for u in u_list:
            df_u = df[df["user_id"] == u]

            q_seq = np.array([q2idx[q] for q in df_u["skill_id"].values])
            r_seq = df_u["correct"].values
            pid_seq = np.array([pid2idx[pid] for pid in df_u["item_id"].values])

            q_seqs.append(q_seq)
            r_seqs.append(r_seq)
            pid_seqs.append(pid_seq)
            negative_r_seqs.append(1 - r_seq)

        return q_seqs, r_seqs, q_list, u_list, r_list, q2idx, u2idx, pid_seqs, pid_list, negative_r_seqs

    def match_seq_len(self, q_seqs, r_seqs, pid_seqs, negative_r_seqs, max_seq_len, pad_val=-1):

        proc_q_seqs = []
        proc_r_seqs = []
        proc_pid_seqs = []
        proc_negative_r_seqs = []

        for q_seq, r_seq, pid_seq, negative_r_seq in zip(q_seqs, r_seqs, pid_seqs, negative_r_seqs):

            i = 0
            while i + max_seq_len < len(q_seq):
                proc_q_seqs.append(q_seq[i:i + max_seq_len])
                proc_r_seqs.append(r_seq[i:i + max_seq_len])
                proc_pid_seqs.append(pid_seq[i:i + max_seq_len])
                proc_negative_r_seqs.append(negative_r_seq[i:i + max_seq_len])

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

        # mask 추출 및 padding 작업
        pad_proc_q_seqs = []
        pad_proc_r_seqs = []
        pad_proc_pid_seqs = []
        pad_proc_negative_r_seqs = []

        for proc_q_seq, proc_r_seq, proc_pid_seq, proc_negative_r_seq in zip(proc_q_seqs, proc_r_seqs, proc_pid_seqs, proc_negative_r_seqs):

            #torch.tensor(aug_q_seq_1, dtype=torch.long)
            pad_proc_q_seqs.append(torch.tensor(proc_q_seq, dtype=torch.long))
            pad_proc_r_seqs.append(torch.tensor(proc_r_seq, dtype=torch.long))
            pad_proc_pid_seqs.append(torch.tensor(proc_pid_seq, dtype=torch.long))
            pad_proc_negative_r_seqs.append(torch.tensor(proc_negative_r_seq, dtype=torch.long))

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

        mask_seqs = (pad_proc_q_seqs != pad_val)

        pad_proc_q_seqs, pad_proc_r_seqs, \
            pad_proc_pid_seqs, pad_proc_negative_r_seqs = \
            pad_proc_q_seqs * mask_seqs, pad_proc_r_seqs * mask_seqs, \
                pad_proc_pid_seqs * mask_seqs, pad_proc_negative_r_seqs * mask_seqs

        return pad_proc_q_seqs, pad_proc_r_seqs, pad_proc_pid_seqs, pad_proc_negative_r_seqs, mask_seqs


    # augmentation을 위한 추가
    def __getitem_internal__(self, index):

        '''
        그냥 train, test 상관없이 augmentation 만들고,
        valid와 test에서는 쓰지 않기
        '''

        t1 = augment_seq_func(
            self.q_seqs[index],
            self.r_seqs[index],
            self.pid_seqs[index],
            self.config
        )

        t2 = augment_seq_func(
            self.q_seqs[index],
            self.r_seqs[index],
            self.pid_seqs[index],
            self.config
        )


        return {
            "concepts": self.q_seqs[index], 
            "responses": self.r_seqs[index], 
            "questions": self.pid_seqs[index], 
            "negative_responses": self.negative_r_seqs[index], 
            "masks": self.mask_seqs[index]
            }

    def __getitem__(self, index):
        return self.__getitem_internal__(index)