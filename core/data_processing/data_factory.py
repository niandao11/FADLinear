import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from core.data_processing.data_loader import (
    Dataset_Custom,
    Dataset_ETT_SAW,
    Dataset_ETT_hour,
    Dataset_ETT_minute,
    Dataset_Pred,
)


DATASETS = {
    "ETTh1": Dataset_ETT_hour,
    "ETTh2": Dataset_ETT_hour,
    "ETTm1": Dataset_ETT_minute,
    "ETTm2": Dataset_ETT_minute,
    "ETTm2_SAW": Dataset_ETT_SAW,
    "weather": Dataset_Custom,
}


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def data_provider(args, flag):
    data_class = Dataset_Pred if flag == "pred" else DATASETS[args.data]
    time_encoding = 0 if args.embed != "timeF" else 1
    if flag == "test":
        shuffle = False
        drop_last = False
        batch_size = args.batch_size
    elif flag == "pred":
        shuffle = False
        drop_last = False
        batch_size = 1
    else:
        shuffle = True
        drop_last = True
        batch_size = args.batch_size
    dataset = data_class(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=time_encoding,
        freq=args.freq,
        train_only=args.train_only,
    )
    print(flag, len(dataset))
    generator = torch.Generator()
    generator.manual_seed(getattr(args, "current_seed", args.random_seed))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        drop_last=drop_last,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    return dataset, loader
