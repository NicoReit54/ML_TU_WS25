import torch 
import torch.nn as nn 
import torch.nn.functional as F

# Few sources:
# https://docs.pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
# https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
# https://www.geeksforgeeks.org/deep-learning/building-a-convolutional-neural-network-using-pytorch/
# https://stackoverflow.com/questions/79228528/i-am-trying-to-create-multiscale-cnn-but-facing-this-error-runtimeerror-mat1


class CNN(nn.Module):
    """
    Convolutional Neural Network
    """

    def __init__(self,
                 in_channels: int = 3, 
                 num_classes: int = 10,
                 input_size: int = 32, 
                 base_channels: int = 32,
                 channel_multiplier: int = 2) -> None:
        '''
        Initializes the CNN model with convolutional layers.

        :param in_channels: How many channels the input images have (e.g., 3 for RGB images).
        :type in_channels: int
        :param num_classes: Number of available output classes for classification.
        :type num_classes: int
        :param input_size: Size (height/width) of the input images.
        :type input_size: int
        :param base_channels: Number of wanted channels in the first convolutional layer.
        :type base_channels: int
        :param channel_multiplier: Factor by which the number of channels increases after each convolutional layer.
        :type channel_multiplier: int
        '''
        super().__init__()

        # precompute the channels and sizes after each conv layer 
        c1 = base_channels 
        c2 = c1 * channel_multiplier 
        c3 = c2 * channel_multiplier

        # Feature extractor 
        self.features = nn.Sequential(
            # Input shape for CIFAR-10: (batch_size, 3, 32, 32)
            nn.Conv2d(in_channels=in_channels, 
                      out_channels=c1, 
                      kernel_size=3, padding=1), # kernel size 3 kind of standard, but also the picutres are not too high-res. With padding=1 this preserves the size

            nn.ReLU(),
            # Subsampling(Pooling) layer: replaces 2D patches by their maximum (“max-pooling”). It reduces the size of
            # feature maps (i.e. here the output of a convolution layer and thus the input to the next layer)
            nn.MaxPool2d(kernel_size=2),  # ie for CIFAR: reduces size by two -> (32, 16, 16)

            nn.Conv2d(in_channels = c1, 
                      out_channels = c2, 
                      kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # ie for CIFAR -> (64, 8, 8)

            nn.Conv2d(in_channels = c2, 
                      out_channels = c3, 
                      kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # ie for CIFAR -> (128, 4, 4)
        )

        # Calculate the size of the flattened feature maps after the convolutional layers to define the first linear layer because
        # the input size of the linear layer depends on the output size of the convolutional layers.
        self.flattened_size = self._get_flattened_size(input_size, in_channels)

        # Classifier part: representing the fully connected layers
        self.classifier = nn.Sequential(
            nn.Linear(self.flattened_size, c3 *2), # this here applies a linear transformation to the incoming data: y = xA^T + b
            nn.ReLU(),
            nn.Dropout(p=0.5), # Dropout randomly disables neurons during training to reduce overfitting, p=0.5 means 50% chance to disable a neuron
            nn.Linear(c3 *2, num_classes)
        )

    def _get_flattened_size(self, input_size: int, in_channels: int) -> int:
        """
        Computes the size of the flattened feature maps after the convolutional layers.

        This is necessary to define the input size of the first fully connected layer.

        :param input_size: Size (height/width) of the input images.
        :type input_size: int
        :param in_channels: Number of channels in the input images.
        :type in_channels: int
        :return: Size of the flattened feature maps.
        :rtype: int
        """

        with torch.no_grad(): # We don't need gradients for this computation so lets speed it up
            # Create a dummy input tensor with batch size 1 to pass the dummy input through the feature extractor
            dummy_input = torch.zeros(1, in_channels, input_size, input_size)
            features = self.features(dummy_input)

        # Flatten the output and get its size
        return features.numel()  # Total number of elements

    def forward(self, x):
        """
        Defines the forward pass of the model so pytorch can compute gradients and train the network.
        Basically combine the above defined layers in the right order.
        """
        # Apply the feature extractor layers
        x = self.features(x)

        # Flatten all dimensions except batch size (it looks like: (batch_size, channels, height, width) before flattening)
        # to prepare for the fully connected layers stage
        x = torch.flatten(x, start_dim=1)

        # Apply the classifier layers
        x = self.classifier(x) 

        return x
