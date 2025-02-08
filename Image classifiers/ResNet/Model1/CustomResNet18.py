import torch.nn as nn
import torchvision.models as models

class CustomResNet18(nn.Module):
    def __init__(self, num_classes, freeze_layers=False):
        super(CustomResNet18, self).__init__()
        self.resnet18 = models.resnet18(weights='IMAGENET1K_V1')
        
        # Adjust the first convolutional layer for 32x32 input
        self.resnet18.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.resnet18.maxpool = nn.Identity()  # Remove maxpool for smaller images
        self.resnet18.layer4 = nn.Identity()
        
        # Freeze layers if specified
        if freeze_layers:
            for param in self.resnet18.parameters():
                param.requires_grad = False
        
        in_features = 256
        self.resnet18.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.resnet18(x)