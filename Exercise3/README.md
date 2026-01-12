# README for Exercise 3
This directory contains the code and data for Exercise 3.2.1 - Image Classification of the Machine Learning course.

## File Structure
- `data/load_cifar.py`: This script contains methods to load the CIFAR-10 dataset.

## Usage
To use the data loading functions, you import the `load_cifar.retrieve_all_cifar` module and call it according to what you need (either "test" or "train"). 
For example:

```python
from data.load_cifar import retrieve_all_cifar

test_dict = retrieve_all_cifar(train_test="test")
train_dict = retrieve_all_cifar(train_test="train")

len(test_dict["labels"]), len(train_dict["labels"])  # Should return (10000, 50000)
```
