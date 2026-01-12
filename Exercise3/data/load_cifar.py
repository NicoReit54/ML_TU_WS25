import pickle
import os

import numpy as np
import torch
from torch.utils.data import Dataset

def unpickle(file) -> dict:
    '''
    Method to retrieve the CIFAR data, based on instructions given on:
    https://www.cs.toronto.edu/~kriz/cifar.html
    
    :param file: Filepath to the cifar batch file
    :return: Dictionary with the following elements:
            data -- a 10000x3072 numpy array of uint8s. Each row of the array stores a 32x32 colour image. 
                    The first 1024 entries contain the red channel values, the next 1024 the green, and the 
                    final 1024 the blue. The image is stored in row-major order, so that the first 32 entries 
                    of the array are the red channel values of the first row of the image.
            labels -- a list of 10000 numbers in the range 0-9. The number at index i indicates the label of 
                      the ith image in the array data
    :rtype: dict
    '''
    with open(file, "rb") as fo:
        dict = pickle.load(fo, encoding="bytes")
    return dict 


def retrieve_all_cifar(train_test: str = "train") -> dict:
    '''
    Method to retrieve all CIFAR data batches and combine them into a single dictionary.

    :param train_test: String indicating whether to load the training set ("train") or the test set ("test")
    :return: Dictionary with the following elements:
            data -- a 50000x3072 numpy array of uint8s. Each row of the array stores a 32x32 colour image. 
                    The first 1024 entries contain the red channel values, the next 1024 the green, and the 
                    final 1024 the blue. The image is stored in row-major order, so that the first 32 entries 
                    of the array are the red channel values of the first row of the image.
            labels -- a list of 50000 numbers in the range 0-9. The number at index i indicates the label of 
                      the ith image in the array data
    :rtype: dict
    '''
    base_dir = os.path.dirname(os.path.abspath(__file__)) 
    cifar_dir = os.path.join(base_dir, "cifar-10-batches-py")

    cifar_data = {
        'data': [],
        'labels': []
    }
    if train_test == "train":
        for i in range(1, 6):
            batch_file = os.path.join(cifar_dir, f'data_batch_{i}')
            batch_data = unpickle(batch_file)
            cifar_data['data'].extend(batch_data[b'data'])
            cifar_data['labels'].extend(batch_data[b'labels'])
    elif train_test == "test":
        batch_file = os.path.join(cifar_dir, f'test_batch')
        batch_data = unpickle(batch_file)
        cifar_data['data'].extend(batch_data[b'data'])
        cifar_data['labels'].extend(batch_data[b'labels'])
    else:
        raise ValueError("""Use either "test" or "train" for train_test variable.""")
    
    return cifar_data

class CIFAR10(Dataset):
    '''
    To make the dataset loader compatible with PyTorch's DataLoader class, we create a custom Dataset class.
    This class loads the CIFAR-10 data with the above defined methods and implements the required methods 
    __len__ and __getitem__.
    '''
    def __init__(self, split: str = "train"):
        data_dict = retrieve_all_cifar(train_test=split)

        # Convert to numpy arrays for efficient reshape/indexing
        self.x = np.asarray(data_dict["data"], dtype=np.uint8)      # (N, 3072) because each vector is stored as a  numpy array of 3072 uint8s
        self.y = np.asarray(data_dict["labels"], dtype=np.int64)    # (N,)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx):
        flat = self.x[idx]  # (3072,)

        # CIFAR layout will be (3, 32, 32) when reshaped this way
        img = torch.from_numpy(flat).view(3, 32, 32).float() / 255.0
        label = torch.tensor(self.y[idx], dtype=torch.long)

        return img, label
