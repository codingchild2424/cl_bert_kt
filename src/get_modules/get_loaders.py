from torch.utils.data import DataLoader, random_split, Subset, ConcatDataset
#from utils import pid_collate_fn, pid_diff_collate_fn

from dataloaders.pid_loader import PID_LOADER
from dataloaders.sim_loader import SIM_LOADER
from dataloaders.sim_diff_loader import SIM_DIFF_LOADER

def get_loaders(config, idx=None):

    num_q = None
    num_r = None
    num_pid = None

    # 1. choose the loaders
    # pid loaders
    if config.dataset_name == "assist09":
        dataset_dir = "../datasets/assist09/preprocessed_df.csv"
    elif config.dataset_name == "assist12":
        dataset_dir = "../datasets/assist12/preprocessed_df.csv"
    elif config.dataset_name == "assist17":
        dataset_dir = "../datasets/assist17/preprocessed_df.csv"
    elif config.dataset_name == "algebra05":
        dataset_dir = "../datasets/algebra05/preprocessed_df.csv"
    elif config.dataset_name == "algebra06":
        dataset_dir = "../datasets/bridge_algebra06/preprocessed_df.csv"
    elif config.dataset_name == "ednet":
        dataset_dir = "../datasets/ednet/preprocessed_df.csv"
    
    # pid_loader or sim_loader
    if config.use_augment:
        dataset = SIM_DIFF_LOADER(config.max_seq_len, dataset_dir, config, idx)
    else:
        dataset = PID_LOADER(config.max_seq_len, dataset_dir)

    num_q = dataset.num_q
    num_r = dataset.num_r
    num_pid = dataset.num_pid
    num_diff = dataset.num_diff
    #collate = dataset.collate_fn

    # 2. data chunk

    # if fivefold = True
    if config.fivefold == True:

        first_chunk = Subset(dataset, range( int(len(dataset) * 0.2) ))
        second_chunk = Subset(dataset, range( int(len(dataset) * 0.2), int(len(dataset)* 0.4) ))
        third_chunk = Subset(dataset, range( int(len(dataset) * 0.4), int(len(dataset) * 0.6) ))
        fourth_chunk = Subset(dataset, range( int(len(dataset) * 0.6), int(len(dataset) * 0.8) ))
        fifth_chunk = Subset(dataset, range( int(len(dataset) * 0.8), int(len(dataset)) ))

        # idx from main
        # fivefold first
        if idx == 0:
            # train_dataset is 0.8 of whole dataset
            train_dataset = ConcatDataset([second_chunk, third_chunk, fourth_chunk, fifth_chunk])
            # valid_size is 0.1 of train_dataset
            valid_size = int( len(train_dataset) * config.valid_ratio)
            # train_size is 0.9 of train_dataset
            train_size = int( len(train_dataset) ) - valid_size
            
            train_dataset, valid_dataset = random_split(
                train_dataset, [ train_size, valid_size ]
            )
            # test_dataset is 0.2 of whole dataset
            test_dataset = first_chunk
        # fivefold second
        elif idx == 1:
            train_dataset = ConcatDataset([first_chunk, third_chunk, fourth_chunk, fifth_chunk])
            valid_size = int( len(train_dataset) * config.valid_ratio) #train의 0.1
            train_size = int( len(train_dataset) ) - valid_size #train의 0.9
            
            train_dataset, valid_dataset = random_split(
                train_dataset, [ train_size, valid_size ]
            )
            test_dataset = second_chunk
        # fivefold third
        elif idx == 2:
            train_dataset = ConcatDataset([first_chunk, second_chunk, fourth_chunk, fifth_chunk])
            valid_size = int( len(train_dataset) * config.valid_ratio) #train의 0.1
            train_size = int( len(train_dataset) ) - valid_size #train의 0.9
            
            train_dataset, valid_dataset = random_split(
                train_dataset, [ train_size, valid_size ]
            )
            test_dataset = third_chunk
        # fivefold fourth
        elif idx == 3:
            train_dataset = ConcatDataset([first_chunk, second_chunk, third_chunk, fifth_chunk])
            valid_size = int( len(train_dataset) * config.valid_ratio) #train의 0.1
            train_size = int( len(train_dataset) ) - valid_size #train의 0.9
            
            train_dataset, valid_dataset = random_split(
                train_dataset, [ train_size, valid_size ]
            )
            test_dataset = fourth_chunk
        # fivefold fifth
        elif idx == 4:
            train_dataset = ConcatDataset([first_chunk, second_chunk, third_chunk, fourth_chunk])
            valid_size = int( len(train_dataset) * config.valid_ratio) #train의 0.1
            train_size = int( len(train_dataset) ) - valid_size #train의 0.9
            
            train_dataset, valid_dataset = random_split(
                train_dataset, [ train_size, valid_size ]
            )
            test_dataset = fifth_chunk
    # fivefold = False
    else:
        train_size = int( len(dataset) * config.train_ratio * (1 - config.valid_ratio))
        valid_size = int( len(dataset) * config.train_ratio * config.valid_ratio)
        test_size = len(dataset) - (train_size + valid_size)

        train_dataset, valid_dataset, test_dataset = random_split(
            dataset, [ train_size, valid_size, test_size ]
            )

    # 3. get DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size = config.batch_size,
        shuffle = True, # train_loader use shuffle
        #collate_fn = collate
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size = config.batch_size,
        shuffle = False, # valid_loader don't use shuffle
        #collate_fn = collate
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size = config.batch_size,
        shuffle = False, # test_loader don't use shuffle
        #collate_fn = collate
    )

    return train_loader, valid_loader, test_loader, num_q, num_r, num_pid, num_diff