from torch.utils.data import Dataset
import os
from collections import defaultdict

class DataLoader(Dataset):
    def __init__(self, df, seq_len, num_skills, num_questions):
        self.df = df
        self.seq_len = seq_len
        self.num_skills = num_skills
        self.num_questions = num_questions

        self.questions = [
            u_df["item_id"].values[-self.seq_len :]
            for _, u_df in self.df.groupby("user_id")
        ]
        self.skills = [
            u_df["skill_id"].values[-self.seq_len :]
            for _, u_df in self.df.groupby("user_id")
        ]
        self.responses = [
            u_df["correct"].values[-self.seq_len :]
            for _, u_df in self.df.groupby("user_id")
        ]
        self.lengths = [
            len(u_df["skill_id"].values) for _, u_df in self.df.groupby("user_id")
        ]

        skill_correct = defaultdict(int)
        skill_count = defaultdict(int)
        for s_list, r_list in zip(self.skills, self.responses):
            for s, r in zip(s_list, r_list):
                skill_correct[s] += r
                skill_count[s] += 1

        skill_difficulty = {
            s: skill_correct[s] / float(skill_count[s]) for s in skill_correct
        }
        ordered_skills = [
            item[0] for item in sorted(skill_difficulty.items(), key=lambda x: x[1])
        ]
        self.easier_skills = {}
        self.harder_skills = {}
        for i, s in enumerate(ordered_skills):
            if i == 0:  # the hardest
                self.easier_skills[s] = ordered_skills[i + 1]
                self.harder_skills[s] = s
            elif i == len(ordered_skills) - 1:  # the easiest
                self.easier_skills[s] = s
                self.harder_skills[s] = ordered_skills[i - 1]
            else:
                self.easier_skills[s] = ordered_skills[i + 1]
                self.harder_skills[s] = ordered_skills[i - 1]

        cnt = 0
        for interactions in self.questions:
            cnt += len(interactions)
        self.num_interactions = cnt

        self.len = len(self.questions)

        self.padded_q = torch.zeros(
            (len(self.questions), self.seq_len), dtype=torch.long
        )
        self.padded_s = torch.zeros((len(self.skills), self.seq_len), dtype=torch.long)
        self.padded_r = torch.full(
            (len(self.responses), self.seq_len), -1, dtype=torch.long
        )
        self.attention_mask = torch.zeros(
            (len(self.skills), self.seq_len), dtype=torch.long
        )

        for i, elem in enumerate(zip(self.questions, self.skills, self.responses)):
            q, s, r = elem
            self.padded_q[i, -len(q) :] = torch.tensor(q, dtype=torch.long)
            self.padded_s[i, -len(s) :] = torch.tensor(s, dtype=torch.long)
            self.padded_r[i, -len(r) :] = torch.tensor(r, dtype=torch.long)
            self.attention_mask[i, -len(s) :] = torch.ones(len(s), dtype=torch.long)

    def __getitem__(self, index):

        return {
            "questions": self.padded_q[index],
            "skills": self.padded_s[index],
            "responses": self.padded_r[index],
            "attention_mask": self.attention_mask[index],
        }

    def __len__(self):
        return self.len