import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from utils import pid_collate_fn

class PID_LOADER(Dataset):
    def __init__(self, max_seq_len, dataset_dir) -> None:
        super().__init__()

        self.dataset_dir = dataset_dir
        
        self.q_seqs, self.r_seqs, self.q_list, self.u_list, self.r_list, self.q2idx, \
            self.u2idx, self.pid_seqs, self.pid_list, self.negative_r_seqs = self.preprocess()

        self.num_u = self.u_list.shape[0]
        self.num_q = self.q_list.shape[0]
        self.num_r = self.r_list.shape[0]
        self.num_pid = self.pid_list.shape[0]
        self.num_diff = None

        self.collate_fn = pid_collate_fn

        self.q_seqs, self.r_seqs, self.pid_seqs, self.negative_r_seqs = \
            self.match_seq_len(self.q_seqs, self.r_seqs, self.pid_seqs, self.negative_r_seqs, max_seq_len)

        self.len = len(self.q_seqs)

    # augmentation을 위한 추가
    # def __getitem_internal__(self, index):
        
    #     if self.eval_mode:
    #         return {
    #             "concepts": (self.q_seqs,),
    #             "questions": (self.pid_seqs,),
    #             "responses": (self.r_seqs,),
    #             "negative_responses": (self.negative_r_seqs,)
    #         }
    #     else:
    #         return {
    #             "concepts": (self.q_seqs,),
    #             "questions": (self.pid_seqs,),
    #             "responses": (self.r_seqs,),
    #             "negative_responses": (self.negative_r_seqs,)
    #         }

    def __getitem__(self, index):
        return self.q_seqs[index], self.r_seqs[index], self.pid_seqs[index], self.negative_r_seqs[index]

    # def __getitem__(self, index):
    #     return self.__getitem_internal__(index)

    def __len__(self):
        return self.len

    def preprocess(self):
        df = pd.read_csv(self.dataset_dir, encoding="ISO-8859-1", sep='\t')
        df = df[(df["correct"] == 0) | (df["correct"] == 1)]

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

        return proc_q_seqs, proc_r_seqs, proc_pid_seqs, proc_negative_r_seqs