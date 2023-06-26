import numpy as np
import pandas as pd
import os
from tqdm import tqdm


import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from utils import pid_collate_fn
from modules.augment_seq import augment_seq_func

# transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import mean_squared_error

##########################################################
# SIM_DIFF_LLM_LOADER
# 1. Using trainging dataset, pre-trained bert was finetuned to predict difficulty.
# 2. Predict the difficulty in the valid and test set, 
# Questions and concept which are not in the training dataset are 
##########################################################
class SIM_DIFF_LLM_LOADER(Dataset):
    def __init__(
            self, 
            max_seq_len, 
            dataset_dir, 
            config, 
            idx=None
            ) -> None:
        super(SIM_DIFF_LLM_LOADER, self).__init__()

        ##########################################################
        # Transformers
        ##########################################################
        self.bert_model_name = config.bert_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.bert_model_name)
        self.bert_model = AutoModelForSequenceClassification.from_pretrained(self.bert_model_name, num_labels=1)

        ##########################################################
        # Normal
        ##########################################################
        self.idx = idx

        self.dataset_dir = dataset_dir

        self.config = config
        
        self.q_seqs, self.r_seqs, self.q_list, self.u_list, self.r_list, self.q2idx, \
            self.u2idx, self.pid_seqs, self.pid_list, self.negative_r_seqs, \
                self.q_diff_seqs, self.pid_diff_seqs, self.negative_q_diff_seqs, self.negative_pid_diff_seqs, \
                    self.q_diff_list, self.pid_diff_list, self.negative_q_diff_list, self.negative_pid_diff_list = self.preprocess()

        self.num_u = self.u_list.shape[0]
        self.num_q = self.q_list.shape[0]
        self.num_r = self.r_list.shape[0]
        self.num_pid = self.pid_list.shape[0]
        self.num_q_diff = self.q_diff_list.shape[0]
        self.num_pid_diff = self.pid_diff_list.shape[0]
        self.num_negative_q_diff = self.negative_q_diff_list.shape[0]
        self.num_negative_pid_diff = self.negative_pid_diff_list.shape[0]

        self.collate_fn = pid_collate_fn

        self.q_seqs, self.r_seqs, self.pid_seqs, self.negative_r_seqs, self.q_diff_seqs, self.pid_diff_seqs, self.negative_q_diff_seqs, self.negative_pid_diff_seqs, self.mask_seqs = \
            self.match_seq_len(self.q_seqs, self.r_seqs, self.pid_seqs, self.negative_r_seqs, self.q_diff_seqs, self.pid_diff_seqs, self.negative_q_diff_seqs, self.negative_pid_diff_seqs, max_seq_len)

        self.len = len(self.q_seqs)


    def __len__(self):
        return self.len

    def preprocess(self):

        ############################################################
        # question_curriculum_link path
        ############################################################
        question_curriculum_link_path = os.path.join(self.dataset_dir, "questions.csv")
        question_curriculum_link_df = pd.read_csv(
            question_curriculum_link_path,
            encoding="utf-8",
            sep='|'
        )

        """
        question_id|question_type|difficulty|behavior_id|curriculum_id
        201144|61|2|1|11101
        """

        ############################################################
        # curriculum_meta
        ############################################################
        curriculum_meta_path = os.path.join(self.dataset_dir, "curriculums.csv")
        curriculum_meta_df = pd.read_csv(
            curriculum_meta_path,
            encoding="utf-8",
            sep='|'
        )

        cur_ids = []
        cur_texts = []

        for row in curriculum_meta_df.iterrows():
            cur_ids.append(row[1]['curriculum_id'])

            cur_text = row[1]['curriculum_l_nm'] + " " + row[1]['curriculum_m_nm']

            cur_texts.append(cur_text)

        pre_c_meta_df = pd.DataFrame({
            'curriculum_id': cur_ids,
            'curriculum_text': cur_texts
        })


        ############################################################
        # question meta
        ############################################################
        q_meta_path = os.path.join(self.dataset_dir, "question_meta.csv")
        q_meta_df = pd.read_csv(
            q_meta_path,
            encoding="utf-8",
            sep='|'
        )

        import re
        def clean_html_tags(text):
            clean = re.compile('<.*?>')
            return re.sub(clean, '', text)
        
        q_meta_df['question_body'] = q_meta_df['question_body'].apply(clean_html_tags).str.replace("&nbsp;", " ")

        pre_q_meta_df = q_meta_df[['question_id', 'question_body']]


        ############################################################
        # user_response data path 
        ############################################################
        user_response_folder_path = os.path.join(self.dataset_dir, "user_questions")
        user_response_file_path_list = [
            os.path.join(
                user_response_folder_path, 
                file_name
                ) for file_name in os.listdir(user_response_folder_path)
        ]

        ############################################################
        # 루프로 돌리기
        ############################################################

        user_ids = []
        skill_ids = []
        item_ids = []
        corrects = []
        eventtimes = [] # for sorting

        skill_texts = []
        item_texts = []

        for idx, user_response_file_path in tqdm(enumerate(user_response_file_path_list)):

            user_response_df = pd.read_csv(
                user_response_file_path,
                encoding="ISO-8859-1",
                sep='|'
            )

            for row in user_response_df.iterrows():

                user_id = row[1]['actor_id']

                item_id = row[1]['question_id']

                skill_id = question_curriculum_link_df[
                    question_curriculum_link_df['question_id'] == item_id
                ]['curriculum_id'].values[0]
                
                correct = row[1]['result']

                # q_meta
                skill_text = pre_c_meta_df[
                    pre_c_meta_df['curriculum_id'] == skill_id
                ]['curriculum_text']

                item_text = pre_q_meta_df[
                    pre_q_meta_df['question_id'] == item_id
                ]['question_body']

                user_ids.append(user_id)
                skill_ids.append(skill_id)
                item_ids.append(item_id)
                corrects.append(correct)
                eventtimes.append(row[1]['eventtime'])

                skill_texts.append(skill_text)
                item_texts.append(item_text)

            ############################################################
            # 나중에 없애기
            ############################################################
            if idx == 100:
                break

        df = pd.DataFrame({
            'user_id': user_ids,
            'skill_id': skill_ids,
            'item_id': item_ids,
            'correct': corrects,
            'eventtime': eventtimes,
            'skill_text': skill_texts,
            'item_text': item_texts
        })

        #print(df.head())

        df = df[(df["correct"] == 0) | (df["correct"] == 1)]

        # sort df using eventtime
        df = df.sort_values(by=['eventtime'])
        print(df.head())

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

        # for text dict
        q_text_list = list(df["skill_text"])
        pid_text_list = list(df["item_text"])
        q_text2idx = {idx: q_text  for idx, q_text in enumerate(q_text_list)}
        pid_text2idx = {idx: pid_text for idx, pid_text in enumerate(pid_text_list)}

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
        train_q_seqs = []
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
                train_q_seqs.extend(q_seq)
                train_pid_seqs.extend(pid_seq)
                train_r_seqs.extend(r_seq)

                # train_df
        train_df = pd.DataFrame(
            zip(train_q_seqs, train_pid_seqs, train_r_seqs), 
            columns = ["q", "pid", "r"]
            )
        
        ############################################################
        # valid, test에는 있지만 train에는 없는 q와 pid 목록 구하기
        ############################################################
        # unique q_seqs
        unique_q_seqs = np.unique(train_q_seqs)
        # unique pid_seqs
        unique_pid_seqs = np.unique(train_pid_seqs)

        # unique train_q_seqs
        unique_train_q_seqs = np.unique(train_df["q"].values)
        # unique train_pid_seqs
        unique_train_pid_seqs = np.unique(train_df["pid"].values)

        # valid, test에는 있지만 train에는 없는 q와 pid 목록 구하기
        not_contained_train_q_seqs_list = np.setdiff1d(unique_q_seqs, unique_train_q_seqs)
        not_contained_train_seqs_list = np.setdiff1d(unique_pid_seqs, unique_train_pid_seqs)

        # print("not_contained_train_q_seqs_list", not_contained_train_q_seqs_list)
        # print("not_contained_train_seqs_list", not_contained_train_seqs_list)

        ############################################################
        # q_diff 
        ############################################################
        train_q_diff = np.round(train_df.groupby('q')['r'].mean() * 100)
        q_diff_list = np.unique(train_df.groupby('pid')['r'].mean())


        ############################################################
        # training 데이터만을 활용하여, q_diff 예측모델 만들기
        ############################################################

        q_texts = []
        diffs = []

        for idx, diff in enumerate(train_q_diff):
            q_text = q_text2idx[idx]

            q_texts.append(q_text)
            diffs.append(diff)

        train_q_text_diff_df = pd.DataFrame(
            zip(q_texts, diffs), 
            columns = ["q_text", "diff"]
            )
        
        print("train_q_text_diff_df", train_q_text_diff_df)

        finetuned_bert_model = self.llm_training(
            df=train_q_text_diff_df,
            src_col="q_text",
            tgt_col="diff")

        print("finetuned_bert_model", finetuned_bert_model)

        # inference

        if not_contained_train_q_seqs_list != []:

            not_contained_q_texts = []

            for idx, not_contained_train_q_seq in enumerate(not_contained_train_q_seqs_list):
                    
                    not_contained_q_text = q_text2idx[not_contained_train_q_seq]
                    not_contained_q_texts.append(not_contained_q_text)

            # inference not_contained_q_texts using finetuned_bert_model
            not_contained_q_diffs = finetuned_bert_model(not_contained_q_texts)



        
        # finetuned_bert_model = self.llm_training(
        #     train_df=
        #     test_df=
        #     )




        q_diff_seqs = []
        negative_q_diff_seqs = [] # for neg contrastive learning

        train_q_list = np.unique(train_q_seqs)

        # negative를 일렬로 받아서, negative_q_diff_list 만들기
        train_negative_q_diff_seqs = []

        for q_seq in q_seqs:

            q_diff_seq = []
            negative_q_diff_seq = []

            for q in q_seq:
                if q not in train_q_list:
                    q_diff_seq.append(float(75)) # <PAD>
                    negative_q_diff_seq.append(float(25))
                    train_negative_q_diff_seqs.append(float(25))
                else:
                    q_diff_seq.append(train_q_diff[q])
                    negative_q_diff_seq.append(100 - train_q_diff[q])
                    train_negative_q_diff_seqs.append(1 - train_q_diff[q])

            q_diff_seqs.append(q_diff_seq)
            negative_q_diff_seqs.append(negative_q_diff_seq)

        negative_q_diff_list = np.unique(train_negative_q_diff_seqs)
        
        ############################################################
        # pid_diff
        ############################################################
        train_pid_diff = np.round(train_df.groupby('pid')['r'].mean() * 100)
        pid_diff_list = np.unique(train_df.groupby('pid')['r'].mean()) 

        pid_diff_seqs = []
        negative_pid_diff_seqs = [] # for neg contrastive learning

        train_pid_list = np.unique(train_pid_seqs)

    
        ############################################################
        # negative를 일렬로 받아서, negative_q_diff_list 만들기
        ############################################################
        train_negative_pid_diff_seqs = []

        for pid_seq in pid_seqs:

            pid_diff_seq = []
            negative_pid_diff_seq = []

            for pid in pid_seq:
                if pid not in train_pid_list:
                    pid_diff_seq.append(float(75)) # <PAD>
                    negative_pid_diff_seq.append(float(25))
                else:
                    pid_diff_seq.append(train_pid_diff[pid])
                    negative_pid_diff_seq.append(100 - train_pid_diff[pid])

            pid_diff_seqs.append(pid_diff_seq)
            negative_pid_diff_seqs.append(negative_pid_diff_seq)

        negative_pid_diff_list = np.unique(train_negative_pid_diff_seqs)

        return q_seqs, r_seqs, q_list, u_list, r_list, q2idx, u2idx, pid_seqs, pid_list, negative_r_seqs, \
            q_diff_seqs, pid_diff_seqs, negative_q_diff_seqs, negative_pid_diff_seqs, q_diff_list, pid_diff_list, negative_q_diff_list, negative_pid_diff_list

    def match_seq_len(self, q_seqs, r_seqs, pid_seqs, negative_r_seqs, q_diff_seqs, pid_diff_seqs, negative_q_diff_seqs, negative_pid_diff_seqs, max_seq_len, pad_val=-1):

        proc_q_seqs = []
        proc_r_seqs = []
        proc_pid_seqs = []
        proc_negative_r_seqs = []
        proc_q_diff_seqs = []
        proc_pid_diff_seqs = []
        proc_negative_q_diff_seqs = []
        proc_negative_pid_diff_seqs = []


        for q_seq, r_seq, pid_seq, negative_r_seq, q_diff_seq, pid_diff_seq, negative_q_diff_seq, negative_pid_diff_seq in zip(q_seqs, r_seqs, pid_seqs, negative_r_seqs, q_diff_seqs, pid_diff_seqs, negative_q_diff_seqs, negative_pid_diff_seqs):

            i = 0
            while i + max_seq_len < len(q_seq):
                proc_q_seqs.append(q_seq[i:i + max_seq_len])
                proc_r_seqs.append(r_seq[i:i + max_seq_len])
                proc_pid_seqs.append(pid_seq[i:i + max_seq_len])
                proc_negative_r_seqs.append(negative_r_seq[i:i + max_seq_len])

                proc_q_diff_seqs.append(q_diff_seq[i:i + max_seq_len])
                proc_pid_diff_seqs.append(pid_diff_seq[i:i + max_seq_len])
                proc_negative_q_diff_seqs.append(negative_q_diff_seq[i:i + max_seq_len])
                proc_negative_pid_diff_seqs.append(negative_pid_diff_seq[i:i + max_seq_len])

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

            proc_q_diff_seqs.append(
                np.concatenate(
                    [
                        q_diff_seq[i:],
                        np.array([pad_val] * (i + max_seq_len - len(q_seq)))
                    ]
                )
            )
            proc_pid_diff_seqs.append(
                np.concatenate(
                    [
                        pid_diff_seq[i:],
                        np.array([pad_val] * (i + max_seq_len - len(q_seq)))
                    ]
                )
            )
            proc_negative_q_diff_seqs.append(
                np.concatenate(
                    [
                        negative_q_diff_seq[i:],
                        np.array([pad_val] * (i + max_seq_len - len(q_seq)))
                    ]
                )
            )
            proc_negative_pid_diff_seqs.append(
                np.concatenate(
                    [
                        negative_pid_diff_seq[i:],
                        np.array([pad_val] * (i + max_seq_len - len(q_seq)))
                    ]
                )
            )

        # mask 추출 및 padding 작업
        pad_proc_q_seqs = []
        pad_proc_r_seqs = []
        pad_proc_pid_seqs = []
        pad_proc_negative_r_seqs = []
        pad_proc_q_diff_seqs = []
        pad_proc_pid_diff_seqs = []
        pad_proc_negative_q_diff_seqs = []
        pad_proc_negative_pid_diff_seqs = []

        for proc_q_seq, proc_r_seq, proc_pid_seq, proc_negative_r_seq, proc_q_diff_seq, proc_pid_diff_seq, proc_negative_q_diff_seq, proc_negative_pid_diff_seq in zip(proc_q_seqs, proc_r_seqs, proc_pid_seqs, proc_negative_r_seqs, proc_q_diff_seqs, proc_pid_diff_seqs, proc_negative_q_diff_seqs, proc_negative_pid_diff_seqs):

            #torch.tensor(aug_q_seq_1, dtype=torch.long)
            pad_proc_q_seqs.append(torch.tensor(proc_q_seq, dtype=torch.long))
            pad_proc_r_seqs.append(torch.tensor(proc_r_seq, dtype=torch.long))
            pad_proc_pid_seqs.append(torch.tensor(proc_pid_seq, dtype=torch.long))
            pad_proc_negative_r_seqs.append(torch.tensor(proc_negative_r_seq, dtype=torch.long))
            pad_proc_q_diff_seqs.append(torch.tensor(proc_q_diff_seq, dtype=torch.long))
            pad_proc_pid_diff_seqs.append(torch.tensor(proc_pid_diff_seq, dtype=torch.long))
            pad_proc_negative_q_diff_seqs.append(torch.tensor(proc_negative_q_diff_seq, dtype=torch.long))
            pad_proc_negative_pid_diff_seqs.append(torch.tensor(proc_negative_pid_diff_seq, dtype=torch.long))

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

        pad_proc_q_diff_seqs = pad_sequence(
            pad_proc_q_diff_seqs, batch_first=True, padding_value=pad_val
        )
        pad_proc_pid_diff_seqs = pad_sequence(
            pad_proc_pid_diff_seqs, batch_first=True, padding_value=pad_val
        )
        pad_proc_negative_q_diff_seqs = pad_sequence(
            pad_proc_negative_q_diff_seqs, batch_first=True, padding_value=pad_val
        )
        pad_proc_negative_pid_diff_seqs = pad_sequence(
            pad_proc_negative_pid_diff_seqs, batch_first=True, padding_value=pad_val
        )

        mask_seqs = (pad_proc_q_seqs != pad_val)

        pad_proc_q_seqs, pad_proc_r_seqs, \
            pad_proc_pid_seqs, pad_proc_negative_r_seqs, \
                pad_proc_q_diff_seqs, pad_proc_pid_diff_seqs, pad_proc_negative_q_diff_seqs, pad_proc_negative_pid_diff_seqs = \
            pad_proc_q_seqs * mask_seqs, pad_proc_r_seqs * mask_seqs, \
                pad_proc_pid_seqs * mask_seqs, pad_proc_negative_r_seqs * mask_seqs, \
                    pad_proc_q_diff_seqs * mask_seqs, pad_proc_pid_diff_seqs * mask_seqs, \
                        pad_proc_negative_q_diff_seqs * mask_seqs, pad_proc_negative_pid_diff_seqs * mask_seqs

        return pad_proc_q_seqs, pad_proc_r_seqs, pad_proc_pid_seqs, pad_proc_negative_r_seqs, \
            pad_proc_q_diff_seqs, pad_proc_pid_diff_seqs, pad_proc_negative_q_diff_seqs, pad_proc_negative_pid_diff_seqs, \
                mask_seqs

    def __getitem__(self, index):
        #return self.__getitem_internal__(index)
        return {
            "concepts": self.q_seqs[index], 
            "responses": self.r_seqs[index], 
            "questions": self.pid_seqs[index], 
            "negative_responses": self.negative_r_seqs[index],
            "q_difficult": self.q_diff_seqs[index],
            "pid_difficult": self.pid_diff_seqs[index],
            "negative_q_difficult": self.negative_q_diff_seqs[index],
            "negative_pid_difficult": self.negative_pid_diff_seqs[index],
            "masks": self.mask_seqs[index]
            }
    
    def llm_training(self, df, src_col, tgt_col):

        # Split data into train and test
        train_df = df.sample(frac=0.8, random_state=1)
        test_df = df.drop(train_df.index)

        # Tokenize train and test datasets
        train_encodings = self.tokenizer([str(i) for i in train_df[src_col].values], truncation=True, padding=True, max_length=512)
        test_encodings = self.tokenizer([str(i) for i in test_df[src_col].values], truncation=True, padding=True, max_length=512)
        # Prepare the train and test datasets
        train_dataset = BertDataset(train_encodings, train_df[tgt_col].tolist())
        test_dataset = BertDataset(test_encodings, test_df[tgt_col].tolist())

        # Define the training arguments
        training_args = TrainingArguments(
            output_dir='./results',          # output directory
            num_train_epochs=3,              # total number of training epochs
            per_device_train_batch_size=16,  # batch size per device during training
            per_device_eval_batch_size=64,   # batch size for evaluation
            warmup_steps=400,                # number of warmup steps for learning rate scheduler
            weight_decay=0.01,               # strength of weight decay
            logging_dir='./logs',            # directory for storing logs
            logging_steps=200,               # log & save weights each logging_steps
            evaluation_strategy="steps",     # evaluate each `logging_steps`
        )

        # Define RMSE as evaluation metric
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = predictions[:, 0]
            return {'rmse': mean_squared_error(labels, predictions, squared=False)}

        model = self.bert_model
        print(model)

        # Initialize the Trainer
        trainer = Trainer(
            model=model,                         # the instantiated Transformers model to be trained
            args=training_args,                  # training arguments, defined above
            train_dataset=train_dataset,         # training dataset
            eval_dataset=test_dataset,           # evaluation dataset
            compute_metrics=compute_metrics,     # the callback that computes metrics of interest
        )

        # Train the model
        trainer.train()

        return model


class BertDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)