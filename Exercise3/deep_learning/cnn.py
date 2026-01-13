import torch 
import torch.nn as nn 

# Few sources:
# https://docs.pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
# https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
# https://www.geeksforgeeks.org/deep-learning/building-a-convolutional-neural-network-using-pytorch/
# https://stackoverflow.com/questions/79228528/i-am-trying-to-create-multiscale-cnn-but-facing-this-error-runtimeerror-mat1
# https://discuss.pytorch.org/t/how-to-create-convnet-for-variable-size-input-dimension-images/1906 / https://medium.com/@benjybo7/7-pytorch-pool-methods-you-should-be-using-495eb00325d6

class SimpleCNN(nn.Module):
    """
    SimpleConvolutional Neural Network with three convolutional layers followed by 
    two fully connected layers for classification.

    REMARK: As of now only suitable for square images (height = width).
    """

    def __init__(self,
                 in_channels: int = 3, 
                 num_classes: int = 10,
                 input_size: int = 32, # used before we decided for GAP to allow for variable input sizes
                 base_channels: int = 32, # choice was somewhat arbitrary
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
        c1 = base_channels # we basically decide how many neurons we want to have after the application of the first layer (applying basically ten kernels/filters)
        c2 = c1 * channel_multiplier 
        c3 = c2 * channel_multiplier

        # Feature extractor 
        self.features = nn.Sequential(
            # E.g. Input shape for CIFAR-10: (batch_size, 3, 32, 32)
            nn.Conv2d(in_channels=in_channels, 
                      out_channels=c1, 
                      kernel_size=3, padding=1), # kernel size 3 kind of standard, but also the picutres are not too high-res. With padding=1 this preserves the size

            nn.ReLU(),
            # Subsampling(Pooling) layer: replaces 2D patches by their maximum (“max-pooling”). It reduces ("folds") the size of
            # feature maps (i.e. here the output of a convolution layer and thus the input to the next layer)
            nn.MaxPool2d(kernel_size=2),  # ie for CIFAR: reduces size by two -> (batch_size, 32, 16, 16)

            nn.Conv2d(in_channels = c1, 
                      out_channels = c2, 
                      kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # ie for CIFAR -> (batch_size, 64, 8, 8)

            nn.Conv2d(in_channels = c2, 
                      out_channels = c3, 
                      kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # ie for CIFAR -> (batch_size, 128, 4, 4)
        )

        self.gap = nn.AdaptiveAvgPool2d((1,1))  # Global Average Pooling to reduce spatial dimensions to 1x1 to make the model flexible to different input sizes and it also reduces the number of parameters to train!
        # Further it doesnt make a difference for classification if we average the values to one single value as classifcation cares about whether features are present not where they occur in the picture.
        # ie for CIFAR -> (batch_size, 128, 1, 1)

        # Classifier part: representing the fully connected layers
        self.classifier = nn.Sequential(
            nn.Linear(in_features=c3, out_features=c3 * 2), # this here applies a linear transformation to the incoming data: y = xA^T + b
            nn.ReLU(),
            nn.Dropout(p=0.5), # Dropout randomly disables neurons during training to reduce overfitting, p=0.5 means 50% chance to disable a neuron
            nn.Linear(in_features=c3 * 2, out_features=num_classes)
            # out_features == num_classes, because we want to have one output per class for classification
            # So, e.g. 10 classes: The output will be a vector of size 10, where each element represents the score for each class.
        )

    def forward(self, x):
        """
        Defines the forward pass of the model so pytorch can compute gradients and train the network.
        Basically combine the above defined layers in the right order.
        """
        # Apply the feature extractor layers
        x = self.features(x)

        # Apply the Global Average Pooling defined above
        x = self.gap(x) # (batch_size, c3, 1, 1) -> c3 for CIFAR: 128

        # Flatten all dimensions except batch size (it looks like: (batch_size, channels, height, width) before flattening)
        # to prepare for the fully connected layers stage
        x = torch.flatten(x, start_dim=1) # (batch_size, c3)

        # Apply the classifier layers
        x = self.classifier(x) # (batch_size, num_classes)

        return x
