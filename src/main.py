import os
import argparse
import numpy as np
import pandas as pd
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torch.optim import SGD, Adam

from dataloader import DataLoader

from define_argparser import define_argparser

def main(config):
    accelerator = Accelerator()
    device = accelerator.device
    
    model_name = config.model_name
    dataset_name = config.dataset_name
    seed = config.seed
    batch_size = config.batch_size
    learning_rate = config.learning_rate
    optimizer = config.optimizer
    max_seq_len = config.max_seq_len

    np.random.seed(seed)
    torch.manual_seed(seed)

    df_path = os.path.join(os.path.join("../dataset_path", dataset_name), "preprocessed_df.csv")

    checkpoint_dir = "../checkpoint"

    # dataloader
    dataset = DataLoader

    

    

    

    pass


if __name__ == "__main__":
    config = define_argparser()

    print(config)

    main(config)
