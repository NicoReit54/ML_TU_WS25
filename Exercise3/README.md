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
You can run the shallow learning evaluation (SVM and Random Forest with Histograms/SIFT) by executing the `shallow_client.py` script. Following parameters can be specified via command line arguments and only `--dataset` is required:
General Arguments
+ `--dataset`: 'GTSRB' or 'CIFAR' (required)
+ `--models`: List of models to run. Options: 'RF_HIST', 'RF_SIFT', 'SVM_HIST', 'SVM_SIFT', or 'ALL' (default: 'ALL')
+ `--sample_frac`: Fraction of data to use (0.0 - 1.0). Use 1.0 for full training (default: 0.1)
+ `--result`: Metric to print. Options: 'accuracy', 'f1', 'precision', 'recall', 'all' (default: 'all')
Hyperparameters (Random Forest)
+ `--n_estimators`: Number of trees in the forest (default: 100)
+ `--max_depth`: Maximum depth of the trees (default: None)
Hyperparameters (SVM)
+ `--C`: Regularization parameter (default: 1.0)
+ `--gamma`: Kernel coefficient. Accepts float or 'scale' (default: 'scale')
```bash
python shallow_client.py --dataset GTSRB --models SVM_SIFT --sample_frac 1.0 --result f1 --C 10.0 --gamma 0.01
```

## Running the CNN Training and Evaluation
You can run the CNN training and evaluation by executing the `run_cnn.py` script. 
Following parameters can be specified via command line arguments and only `--dataset` is required:
+ `--dataset`: 'GTSRB' or 'CIFAR' (required)
+ `--epochs`: Number of training epochs (default: 10)
+ `--channel_multiplier`: Channel growth factor per conv layer (default: 2)
+  `--base_channels`: Initial channels in first conv layer (default: 32)
+ `--dropout_rate`: Dropout rate for regularization (default: 0.5)
+ `--kernel_size`: Convolution kernel size (default: 3)
+ `--padding`: Padding for convolutions (default: 1)
+ `--batch_size`: Batch size for dataloaders (default: 128)
+ `--data_augmentation`: Enable/disable augmentation (default: True)
+  `--validation`: Enable/disable validation output (default: True)

### Example (Please mind to run this from within the Exercise3 directory!)
```bash
python run_cnn.py --epochs 20 --dataset CIFAR --channel_multiplier 2 --base_channels 32 --dropout_rate 0.1 --kernel_size 3 --padding 1 --batch_size 128 --data_augmentation True --validation True
```

## Running the Transfer Learning
You can run the (EfficientNet‑B0) training and evaluation by executing the `run_transfer_learning.py` script.  
Following parameters can be specified via command line arguments and only `--dataset` is required:

### Dataset and Dirs
+ `--dataset`: `'cifar10'` or `'gtsrb'` (required)  
+ `--data-dir`: Path to dataset directory (default: `./data`)  
+ `--sample-fraction`: Fraction of GTSRB data to use (default: `1.0`)

### Model Arguments
+ `--dropout`: Dropout rate for classifier head (default: `0.3`)  
+ `--pretrained`: Use ImageNet pretrained weights (default: `True`)  
+ `--no-pretrained`: Disable pretrained weights and train from scratch

### Training Arguments
+ `--batch-size`: Batch size for training (default: `64`)  
+ `--epochs-frozen`: Epochs with frozen EfficientNet backbone (default: `5`)  
+ `--epochs-unfrozen`: Epochs with unfrozen layers for fine‑tuning (default: `10`)  
+ `--lr-frozen`: Learning rate for frozen stage (default: `1e-3`)  
+ `--lr-unfrozen`: Learning rate for unfrozen stage (default: `1e-4`)  
+ `--num-workers`: Number of dataloader workers (default: `2`)

### Output Arguments
+ `--save-dir`: Directory to store checkpoints and plots (default: `./outputs`)  
+ `--model-name`: Custom filename for the saved model (default: auto‑generated)

### Evaluation Arguments
+ `--eval-only`: Run evaluation without training  
+ `--load-model`: Path to a model checkpoint to load

### Device & Reproducibility
+ `--device`: `'cuda'`, `'cpu'`, `'mps'`, or `'auto'` (default: `auto`)  
+ `--seed`: Random seed (default: `42`)


### Example (Please mind to run this from within the Exercise3 directory!)
```bash
python train.py --dataset cifar10  --batch-size 64 --epochs-frozen 5 --epochs-unfrozen 10 --lr-frozen 1e-3 --lr-unfrozen 1e-4 --device auto
```

## Usage of the helper methods
To use the data loading functions, you e.g. import the `load_cifar.retrieve_all_cifar` module and call it according to what you need (either "test" or "train"). Or use the `CIFAR10`/`GTSRB` classes  directly to create an object that is already compatible with PyTorch's DataLoader.

For example for CIFAR-10:
```python
from data.load_cifar import retrieve_all_cifar, CIFAR10
from torch.utils.data import DataLoader

test_dict = retrieve_all_cifar(train_test="test")
train_dict = retrieve_all_cifar(train_test="train")

len(test_dict["labels"]), len(train_dict["labels"])  # Should return (10000, 50000)

train_ds = CIFAR10(split="train")
test_ds = CIFAR10(split="test")

trainloader = DataLoader(train_ds, batch_size=128)
testloader = DataLoader(test_ds, batch_size=256)
```

or for GTSRB:
```python
from data.load_gtsrb import GTSRB
from torch.utils.data import DataLoader

# Define paths to the GTSRB dataset
training_path = "../data/GTSRB/Final_Training/Images"
testing_path = "../data/GTSRB/Final_Test/Images"

# Create datasets and show the Dataloader integration
train_dataset = GTSRB(base_path=training_path, split="train", transform=None)
test_dataset  = GTSRB(base_path=testing_path, split="test", transform=None)

trainloader = DataLoader(train_dataset, batch_size=128)
testloader = DataLoader(test_dataset, batch_size=256)
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
               device=device,
               title="Model Evaluation Title")
```