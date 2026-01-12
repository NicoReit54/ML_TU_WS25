# README for Exercise 3
This directory contains the code and data for Exercise 3.2.1 - Image Classification of the Machine Learning course.

## File Structure
- `data/load_cifar.py`: This script contains methods to load the CIFAR-10 dataset.

## Usage
To use the data loading functions, you import the `load_cifar.retrieve_all_cifar` module and call it according to what you need (either "test" or "train"). Or use the `CIFAR10` class directly to create an object compatible with PyTorch's DataLoader.
For example:

```python
from data.load_cifar import retrieve_all_cifar
from torch.utils.data import DataLoader

test_dict = retrieve_all_cifar(train_test="test")
train_dict = retrieve_all_cifar(train_test="train")

len(test_dict["labels"]), len(train_dict["labels"])  # Should return (10000, 50000)

from data.load_cifar import CIFAR10

train_ds = CIFAR10(split="train")
test_ds = CIFAR10(split="test")

trainloader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
testloader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)

```
