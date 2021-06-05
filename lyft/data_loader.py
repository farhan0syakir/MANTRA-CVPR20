from torch.utils.data import Dataset
import numpy as np
import os


class LyftDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data['image'][index], self.data['history_positions'][index], self.data['target_positions'][index]

    def __len__(self):
        return len(self.data['image'])

def load_files(path, train_or_test):
    aggr = None
    len_train = 0
    for file in os.listdir(path):
        if file.startswith(f"{train_or_test}_"):
            len_train += 1

    for i in range(len_train):
        print(f"load {train_or_test}_{i}")
        with open(f'{path}/{train_or_test}_{i}.npy', 'rb') as f:
            file_data = np.load(f, allow_pickle=True).item()

        if aggr is None:
            aggr = file_data
        else:
            aggr['image'] = np.vstack([aggr['image'], file_data['image']])
            aggr['history_positions'] = np.vstack([aggr['history_positions'], file_data['history_positions']])
            aggr['target_positions'] = np.vstack([aggr['target_positions'], file_data['target_positions']])

    return aggr

def load_data(path):
    train = load_files(path, 'train')
    test = load_files(path, 'test')

    # with open(f'{path}/eval_lyft.npy', 'rb') as f:
    #     test = np.load(f, allow_pickle=True)

    return train, test