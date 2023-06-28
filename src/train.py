import numpy as np
import datetime

import torch
from get_modules.get_loaders import get_loaders
from get_modules.get_models import get_models
from get_modules.get_trainers import get_trainers
from utils import get_optimizers, get_crits, recorder, visualizer

from define_argparser import define_argparser

def main(config, train_loader=None, valid_loader=None, test_loader=None, num_q=None, num_r=None, num_pid=None, num_q_diff=None, num_pid_diff=None, num_negative_q_diff=None, num_negative_pid_diff=None):
    # 0. device setting

    device = torch.device('cpu') if config.gpu_id < 0 else torch.device('cuda:%d' % config.gpu_id)

    print("**********************")
    print("DEVICE", device)
    print("**********************")

    # 1. get dataset from loader
    # 1-1. use fivefold
    if config.fivefold == True:
        train_loader = train_loader
        valid_loader = valid_loader
        test_loader = test_loader
        num_q = num_q
        num_r = num_r
        num_pid = num_pid
        num_q_diff = num_q_diff
        num_pid_diff = num_pid_diff
        num_negative_q_diff = num_negative_q_diff
        num_negative_pid_diff = num_negative_pid_diff
    # 1-2. not use fivefold
    else:
        idx = 0
        train_loader, valid_loader, test_loader, num_q, num_r, num_pid, num_q_diff, num_pid_diff, num_negative_q_diff, num_negative_pid_diff = get_loaders(config, device, idx)

    # 2. select models using get_models
    model = get_models(num_q, num_r, num_pid, num_q_diff, num_pid_diff, num_negative_q_diff, num_negative_pid_diff, device, config)
    
    # 3. select optimizers using get_optimizers
    optimizer = get_optimizers(model, config)
    
    # 4. select crits using get_crits
    crit = get_crits(config)
    
    # 5. select trainers for models, using get_trainers
    trainer = get_trainers(model, optimizer, device, num_q, num_pid, crit, config)

    # 6. use trainer.train to train the models
    # the result contain train_scores, valid_scores, hightest_valid_score, highest_test_score
    train_auc_scores, valid_aue_scores, test_auc_scores, \
        train_rmse_scores, valid_rmse_scores, test_rmse_scores, \
            best_auc_test_score, best_rmse_test_score = trainer.train(train_loader, valid_loader, test_loader, config)

    # 7. model record
    # for model's name
    today = datetime.datetime.today()
    record_time = str(today.month) + "_" + str(today.day) + "_" + str(today.hour) + "_" + str(today.minute)
    # model's path
    model_path = '../model_records/' + str(best_auc_test_score) + "_" + record_time + "_" + config.model_fn
    # model save
    torch.save({
        'model': trainer.model.state_dict(),
        'config': config
    }, model_path)

    return train_auc_scores, valid_aue_scores, test_auc_scores, \
        train_rmse_scores, valid_rmse_scores, test_rmse_scores, \
            best_auc_test_score, best_rmse_test_score, record_time

# If you used python train.py, then this will be start first
if __name__ == "__main__":
    # get config from define_argparser
    config = define_argparser()

    device = torch.device('cpu') if config.gpu_id < 0 else torch.device('cuda:%d' % config.gpu_id)

    # if fivefold = True
    if config.fivefold == True:

        test_auc_scores_list = []
        test_rmse_scores_list = []
        
        for idx in range(5):
            train_loader, valid_loader, test_loader, num_q, num_r, num_pid, num_q_diff, num_pid_diff, num_negative_q_diff, num_negative_pid_diff = get_loaders(config, device, idx)
            train_auc_scores, valid_aue_scores, test_auc_scores, \
                train_rmse_scores, valid_rmse_scores, test_rmse_scores, \
                    best_auc_test_score, best_rmse_test_score, record_time= main(config, train_loader, valid_loader, test_loader, num_q, num_r, num_pid, num_q_diff, num_pid_diff, num_negative_q_diff, num_negative_pid_diff)
            test_auc_scores_list.append(best_auc_test_score)
            test_rmse_scores_list.append(best_rmse_test_score)

        # mean the test_scores_list
        test_auc_score_mean = sum(test_auc_scores_list)/5
        test_rmse_score_mean = sum(test_rmse_scores_list)/5
        # for record
        recorder(test_auc_score_mean, test_rmse_score_mean, record_time, config)
    # if fivefold = False 
    else:
        train_auc_scores, valid_aue_scores, test_auc_scores, \
                train_rmse_scores, valid_rmse_scores, test_rmse_scores, \
                    best_auc_test_score, best_rmse_test_score, record_time = main(config)
        # for record
        recorder(best_auc_test_score, best_rmse_test_score, record_time, config)
        # for visualizer
        #visualizer(train_auc_scores, valid_auc_scores, record_time)
    