# README for Exercise 3
This directory contains the code and data for Exercise 3.2.1 - Image Classification of the Machine Learning course.

## Project Structure
```
Exercise3/
│── data/
│   ├── load_cifar.py               # Module for loading and preprocessing the CIFAR-10 dataset
│   └── load_gtsrb.py               # Module for loading and preprocessing the GTSRB dataset
│── deep_learning/
│   ├── cnn.py                      # Implementation of the Convolutional Neural Network (CNN) model
│   ├── train.py                    # Module for training the CNN model
│   ├── train_transfer_learning.py  # Module for transfer learning using pre-trained models
│   └── cnn_transfer_learning.py    # Implementation of transfer learning with CNNs
│── shallow_ml/
│   ├── shallow_client.py           # Implementation of a shallow machine learning model
│── analysis/
│   └── evaluation.py               # Module for evaluating model performance
│── run_cnn.py                      # Script to train and evaluate the CNN model
│── run_transfer_learning.py        # Script to perform transfer learning using a pre-trained model
│── requirements.txt                # List of required Python packages
│── README.md                       # This README file

```

## Requirements
To run the code, you need to have Python >= 3.12.12 installed along with the required packages (which you can install using pip):
```bash
pip install -r requirements.txt
```

## Running the Shallow Model
TBD

## Running the CNN Training and Evaluation
You can run the CNN training and evaluation by executing the `run_cnn.py` script. 
Following parameters can be specified via command line arguments and only `dataset` is required:
+    --dataset: 'GTSRB' or 'CIFAR' (required)
+    --epochs: Number of training epochs (default: 10)
+    --channel_multiplier: Channel growth factor per conv layer (default: 2)
+    --base_channels: Initial channels in first conv layer (default: 32)
+    --dropout_rate: Dropout rate for regularization (default: 0.5)
+    --kernel_size: Convolution kernel size (default: 3)
+    --padding: Padding for convolutions (default: 1)
+    --batch_size: Batch size for dataloaders (default: 128)
+    --data_augmentation: Enable/disable augmentation (default: True)
+    --validation: Enable/disable validation output (default: True)

```bash
python run_cnn.py --epochs 20 --dataset CIFAR --channel_multiplier 2 --base_channels 32 --dropout_rate 0.1 --kernel_size 3 --padding 1 --batch_size 128 --data_augmentation True --validation True
```

## Running the Transfer Learning
To run transfer learning using a pre-trained model, you can use the `run_transfer_learning.py` script. Similar command line arguments can be specified as in `run_cnn.py`, with the addition of
TBD



## Usage of the helper methods
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

## Evaluation
To evaluate the model, we have set up the `evaluate_model` function that can be pulled from the `analysis.evaluation` module. This method takes an already trained model and a DataLoader (with the validation/test set) as input and returns ... :    
- the overall accuracy
- per-class accuracy
- also precision / recall / f1-scores
- and visualizes the confusion matrix
... of the model on the provided dataset.

```python
from analysis.evaluation import evaluate_model  
evaluate_model(model=trained_model,
               dataloader=testloader,
               class_names=["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"],
               figsize=(10, 8),
               device="cpu",
               title="Model Evaluation Title")
```