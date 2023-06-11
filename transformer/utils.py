import torch
from .dataset import train_data, val_data, stoi, itos
from .hyperparameters import batch_size, block_size

torch.manual_seed(1337)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def encode(s): return [stoi[c] for c in s]
def decode(l): return ''.join([itos[i] for i in l])

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))

    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])

    x, y = x.to(device), y.to(device)

    return x, y
