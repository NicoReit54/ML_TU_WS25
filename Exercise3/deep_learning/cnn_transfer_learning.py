import torch
import torch.nn as nn
from torchvision import models


class EfficientNetTransferModel(nn.Module):
    """
    EfficientNet-B0 with custom classifier head for transfer learning.
    Supports freezing/unfreezing backbone layers for two-stage training.
    """
    
    def __init__(self, num_classes: int, dropout: float = 0.3, pretrained: bool = True):
        """
        Initialize the transfer learning model.
        
        :param num_classes: Number of output classes
        :param dropout: Dropout rate for the classifier head
        :param pretrained: Whether to use ImageNet pretrained weights
        """
        super(EfficientNetTransferModel, self).__init__()
        
        # Load pretrained EfficientNet-B0
        if pretrained:
            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
            self.backbone = models.efficientnet_b0(weights=weights)
        else:
            self.backbone = models.efficientnet_b0(weights=None)
        
        # Get input features for classifier
        in_features = self.backbone.classifier[1].in_features
        
        # Replace classifier with custom head
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features, num_classes)
        )
    
    def forward(self, x):
        """Forward pass through the model"""
        return self.backbone(x)
    
    def freeze_backbone(self):
        """Freeze all layers except the classifier head"""
        for param in self.backbone.features.parameters():
            param.requires_grad = False
        
        # Ensure classifier is trainable
        for param in self.backbone.classifier.parameters():
            param.requires_grad = True
    
    def unfreeze_backbone(self, num_layers: int = 2):
        """
        Unfreeze the last num_layers blocks of the backbone for fine-tuning.
        
        :param num_layers: Number of blocks to unfreeze from the end
        """
        total_blocks = len(self.backbone.features)
        unfreeze_from = max(0, total_blocks - num_layers)
        
        for i, block in enumerate(self.backbone.features):
            if i >= unfreeze_from:
                for param in block.parameters():
                    param.requires_grad = True
    
    def get_trainable_params(self):
        """Get count of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_total_params(self):
        """Get count of total parameters"""
        return sum(p.numel() for p in self.parameters())
