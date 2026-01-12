import pickle
import os

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