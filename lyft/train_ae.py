import sys
sys.path.append('..')

import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from tqdm import tqdm

from lyft.data_loader import LyftDataset
import argparse
from lyft.trainer.train_ae import Trainer


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=int, default=0.0001)
    parser.add_argument("--max_epochs", type=int, default=600)

    parser.add_argument("--past_len", type=int, default=20, help="length of past (in timesteps)")
    parser.add_argument("--future_len", type=int, default=40, help="length of future (in timesteps)")
    parser.add_argument("--dim_embedding_key", type=int, default=48)

    parser.add_argument("--dataset_file", default="data", help="dataset file")
    parser.add_argument("--info", type=str, default='', help='Name of training. '
                                                             'It will be used in tensorboard log and test folder')
    return parser.parse_args()


def main(config):
    t = Trainer(config)
    print('start training autoencoder')
    t.fit()


# def main_():
#     train, test = load_data('data')
#     train_dataset = LyftDataset(train)  # create your datset
#     train_loader = DataLoader(train_dataset, batch_size=12)  # create your dataloader
#     tr_it = iter(train_loader)
#     progress_bar = tqdm(range(4))
#     for i in progress_bar:
#         try:
#             data = next(tr_it)
#         except StopIteration:
#             tr_it = iter(train_loader)
#             data = next(tr_it)
#         print(data['image'].shape)



if __name__ == '__main__':
    config = parse_config()
    main(config)