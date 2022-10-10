import torch
from copy import deepcopy

from torch.nn.functional import one_hot
from sklearn import metrics
import numpy as np
from tqdm import tqdm
from random import random, randint

from utils import EarlyStopping

from sklearn.metrics import mean_squared_error

from modules.mlm import Mlm4BertTrain, Mlm4BertTest
from modules.augment_seq import augment_seq_func

class CL_MonaCoBERT_Trainer():

    def __init__(
        self, 
        model, 
        optimizer, 
        n_epochs, 
        device, 
        num_q, 
        crit, 
        max_seq_len,
        config,
        grad_acc=False, 
        grad_acc_iter=4
        ):
        self.model = model
        self.optimizer = optimizer
        self.n_epochs = n_epochs
        self.device = device
        self.num_q = num_q
        self.crit = crit
        self.max_seq_len = max_seq_len
        self.grad_acc = grad_acc #gradient accumulation
        self.grad_acc_iter = grad_acc_iter

        self.config = config

        # bce_loss_fn, rmse_loss_fn, ce_loss_fn
        self.binary_cross_entropy = crit[0]
        self.rmse_loss_fn = crit[1]
        self.ce_loss_fn = crit[2]
    
    def _train(self, train_loader):

        auc_score = 0
        y_trues, y_scores = [], []
        loss_list = []

        for idx, batch in enumerate(tqdm(train_loader)):
            self.model.train()

            q_seqs = batch["concepts"].to(self.device)
            r_seqs = batch["responses"].to(self.device)
            pid_seqs = batch["questions"].to(self.device)
            negative_r_seqs = batch["negative_responses"].to(self.device)
            mask_seqs = batch["masks"].to(self.device)

            # for correct
            real_seqs = r_seqs.clone()

            ################
            # BERT MASKING #
            ################
            # mlm_r_seqs: for MLM, [MASK] position get 2 / mlm_idx are index of [MASK]
            mlm_r_seqs, mlm_idxs = Mlm4BertTrain(r_seqs, mask_seqs)
            mlm_r_seqs = mlm_r_seqs.to(self.device)
            mlm_idxs = mlm_idxs.to(self.device)
            # |mlm_r_seqs| = (bs, n)
            # |mlm_idxs| = (bs, n)

            ########################
            # Augmentation Masking #
            ########################

            if self.config.use_augment:
                # 여기에서 augment_seq를 활용해서 각 요소를 도출해야 함
                aug_q_i = q_seqs
                aug_q_j = q_seqs
                aug_r_i = r_seqs
                aug_r_j = r_seqs
                aug_pid_i = pid_seqs
                aug_pid_j = pid_seqs
                mask_i = mask_seqs
                mask_j = mask_seqs
            else:
                # augmentation을 쓰지 않는 경우
                aug_q_i = q_seqs
                aug_q_j = q_seqs
                aug_r_i = r_seqs
                aug_r_j = r_seqs
                aug_pid_i = pid_seqs
                aug_pid_j = pid_seqs
                mask_i = mask_seqs
                mask_j = mask_seqs

            output = self.model(
                q_seqs.long(), 
                mlm_r_seqs.long(), # r_seqs with MLM
                pid_seqs.long(),
                negative_r_seqs.long(),
                mask_seqs.long(), # for attn_mask
                aug_q_i,
                aug_q_j,
                aug_r_i,
                aug_r_j,
                aug_pid_i,
                aug_pid_j,
                mask_i,
                mask_j
            )

            y_hat = output[0].to(self.device) # |y_hat| = (bs, n, output_size=1)
            inter_cos_sim = output[1].to(self.device)
            inter_labels = output[2].to(self.device)
            
            # Original Loss
            y_hat = y_hat.squeeze()
            # |y_hat| = (bs, n)
            y_hat = torch.masked_select(y_hat, mlm_idxs)
            #|y_hat| = (bs * n - n_mlm_idxs)
            correct = torch.masked_select(real_seqs, mlm_idxs)
            #|correct| = (bs * n - n_mlm_idxs)

            correct = torch.tensor(correct, dtype=torch.float)

            bce_loss = self.binary_cross_entropy(y_hat, correct)
            # |loss| = (1)

            # Contrastive Loss
            cl_loss = torch.mean(self.ce_loss_fn(inter_cos_sim, inter_labels))

            # Loss = Original Loss + lambda * Contrastive Loss
            loss = (1 - self.config.cl_lambda) * bce_loss + self.config.cl_lambda * cl_loss

            #loss
            # grad_accumulation = True
            if self.grad_acc == True:
                loss.backward()
                if (idx + 1) % self.grad_acc_iter == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            # grad_accumulation = False
            else:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            y_trues.append(correct)
            y_scores.append(y_hat)
            loss_list.append(loss)

        y_trues = torch.cat(y_trues).detach().cpu().numpy()
        y_scores = torch.cat(y_scores).detach().cpu().numpy()

        auc_score += metrics.roc_auc_score( y_trues, y_scores )
        rmse_score = np.sqrt(mean_squared_error(y_true=y_trues, y_pred=y_scores))
        #rmse_score = torch.mean( torch.Tensor(loss_list) ).detach().cpu().numpy()

        return auc_score, rmse_score

    def _validate(self, valid_loader):

        auc_score = 0
        y_trues, y_scores = [], []
        loss_list = []

        with torch.no_grad():
            for batch in tqdm(valid_loader):
                self.model.eval()
                
                q_seqs = batch["concepts"].to(self.device)
                r_seqs = batch["responses"].to(self.device)
                pid_seqs = batch["questions"].to(self.device)
                negative_r_seqs = batch["negative_responses"].to(self.device)
                mask_seqs = batch["masks"].to(self.device)

                real_seqs = r_seqs.clone()

                mlm_r_seqs, mlm_idxs = Mlm4BertTest(r_seqs, mask_seqs)

                mlm_r_seqs = mlm_r_seqs.to(self.device)
                mlm_idxs = mlm_idxs.to(self.device)

                y_hat = self.model(
                    q_seqs.long(),
                    mlm_r_seqs.long(),
                    pid_seqs.long(),
                    negative_r_seqs.long(),
                    mask_seqs.long()
                ).to(self.device)

                y_hat = y_hat.squeeze()

                y_hat = torch.masked_select(y_hat, mlm_idxs)
                correct = torch.masked_select(real_seqs, mlm_idxs)

                correct = torch.tensor(correct, dtype=torch.float)

                loss = self.binary_cross_entropy(y_hat, correct)

                y_trues.append(correct)
                y_scores.append(y_hat)
                loss_list.append(loss)

        y_trues = torch.cat(y_trues).detach().cpu().numpy()
        y_scores = torch.cat(y_scores).detach().cpu().numpy()

        auc_score += metrics.roc_auc_score( y_trues, y_scores )
        rmse_score = np.sqrt(mean_squared_error(y_true=y_trues, y_pred=y_scores))
        #loss_result = torch.mean(torch.Tensor(loss_list)).detach().cpu().numpy()

        return auc_score, rmse_score

    def _test(self, test_loader):

        auc_score = 0
        y_trues, y_scores = [], []
        loss_list = []

        with torch.no_grad():
            for batch in tqdm(test_loader):
                self.model.eval()

                q_seqs = batch["concepts"].to(self.device)
                r_seqs = batch["responses"].to(self.device)
                pid_seqs = batch["questions"].to(self.device)
                negative_r_seqs = batch["negative_responses"].to(self.device)
                mask_seqs = batch["masks"].to(self.device)

                real_seqs = r_seqs.clone()

                mlm_r_seqs, mlm_idxs = Mlm4BertTest(r_seqs, mask_seqs)

                mlm_r_seqs = mlm_r_seqs.to(self.device)
                mlm_idxs = mlm_idxs.to(self.device)

                y_hat = self.model(
                    q_seqs.long(),
                    mlm_r_seqs.long(),
                    pid_seqs.long(),
                    negative_r_seqs.long(),
                    mask_seqs.long()
                ).to(self.device)

                y_hat = y_hat.squeeze()

                y_hat = torch.masked_select(y_hat, mlm_idxs)
                correct = torch.masked_select(real_seqs, mlm_idxs)

                correct = torch.tensor(correct, dtype=torch.float)

                loss = loss = self.binary_cross_entropy(y_hat, correct)

                y_trues.append(correct)
                y_scores.append(y_hat)
                loss_list.append(loss)

        y_trues = torch.cat(y_trues).detach().cpu().numpy()
        y_scores = torch.cat(y_scores).detach().cpu().numpy()

        auc_score += metrics.roc_auc_score( y_trues, y_scores )
        rmse_score = np.sqrt(mean_squared_error(y_true=y_trues, y_pred=y_scores))
        #loss_result = torch.mean(torch.Tensor(loss_list)).detach().cpu().numpy()

        return auc_score, rmse_score

    # train use the _train, _validate, _test
    def train(self, train_loader, valid_loader, test_loader, config):
        
        best_auc_test_score = 0
        best_auc_valid_score = 0 # for EarlyStopping
        best_rmse_test_score = float('inf')

        train_auc_scores = []
        valid_auc_scores = []
        test_auc_scores = []

        train_rmse_scores = []
        valid_rmse_scores = []
        test_rmse_scores = []

        # early_stopping
        early_stopping = EarlyStopping(best_score=best_auc_valid_score)

        # Train and Valid Session
        for epoch_index in range(self.n_epochs):
            
            print("Epoch(%d/%d) start" % (
                epoch_index + 1,
                self.n_epochs
            ))

            # Training Session
            train_auc_score, train_rmse_score = self._train(train_loader)
            valid_auc_score, valid_rmse_score = self._validate(valid_loader)
            test_auc_score, test_rmse_score = self._test(test_loader)

            # train, test record 저장
            train_auc_scores.append(train_auc_score)
            valid_auc_scores.append(valid_auc_score)
            test_auc_scores.append(test_auc_score)

            train_rmse_scores.append(train_rmse_score)
            valid_rmse_scores.append(valid_rmse_score)
            test_rmse_scores.append(test_rmse_score)

            # early stop
            train_scores_avg = np.average(train_auc_scores)
            valid_scores_avg = np.average(valid_auc_scores)
            early_stopping(valid_scores_avg, self.model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if test_auc_score >= best_auc_test_score:
                best_auc_test_score = test_auc_score
            
            if test_rmse_score <= best_rmse_test_score:
                best_rmse_test_score = test_rmse_score

            print("Epoch(%d/%d) result: train_score=%.4f(%.4f) valid_score=%.4f(%.4f) test_score=%.4f(%.4f) best_test_score=%.4f(%.4f)" % (
                epoch_index + 1,
                self.n_epochs,
                train_auc_score,
                train_rmse_score,
                valid_auc_score,
                valid_rmse_score,
                test_auc_score,
                test_rmse_score,
                best_auc_test_score,
                best_rmse_test_score
            ))

        print("\n")
        print("The Best Test Score in Testing Session is %.4f(%.4f)" % (
                best_auc_test_score,
                best_rmse_test_score
            ))
        print("\n")
        
        self.model.load_state_dict(torch.load("../checkpoints/checkpoint.pt"))

        return train_auc_scores, valid_auc_scores, test_auc_scores, \
            train_rmse_scores, valid_rmse_scores, test_rmse_scores, \
            best_auc_test_score, best_rmse_test_score