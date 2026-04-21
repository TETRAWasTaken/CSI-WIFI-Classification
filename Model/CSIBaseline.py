import torch 
import torch.nn as nn
import torchvision.models as models

class CSIBaseline(nn.Module):
    def __init__(self, num_classes=7):
        super(CSIBaseline, self).__init__()
        
        self.resnet = models.resnet18(weights=None)
        
        original_conv1 = self.resnet.conv1
        self.resnet.conv1 = nn.Conv2d(
            in_channels=1, 
            out_channels=original_conv1.out_channels, 
            kernel_size=original_conv1.kernel_size, 
            stride=original_conv1.stride, 
            padding=original_conv1.padding, 
            bias=original_conv1.bias
        )
        
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.resnet(x)