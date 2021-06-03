from torch.utils.data import Dataset
import numpy as np

class LyftDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data['image'][index], self.data['history_positions'][index], self.data['target_positions'][index]

    def __len__(self):
        return len(self.data['image'])


def load_data(path):

    with open(f'{path}/train_lyft.npy', 'rb') as f:
        train = np.load(f, allow_pickle=True)

    with open(f'{path}/eval_lyft.npy', 'rb') as f:
        test = np.load(f, allow_pickle=True)

    return train.item(), test.item()