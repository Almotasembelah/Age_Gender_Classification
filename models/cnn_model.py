from torch.utils.checkpoint import checkpoint
from torch import nn
from models.models import Classifier, Task

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
                                   padding=padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.relu(x)

class Block(nn.Module):
    def __init__(self, in_filters, out_filters, kernel_size, skip=False):
        super().__init__()
        self.skip = skip
        padding = (kernel_size - 1) // 2
        self.conv = nn.Sequential(
            DepthwiseSeparableConv(in_filters, out_filters, kernel_size, padding),
            nn.BatchNorm2d(out_filters),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return 0.95 * self.conv(x) + x if self.skip else self.conv(x)

class ModelV1(nn.Module):
    def __init__(self):
        super().__init__(task=Task.ALL)
        self.conv5X5 = nn.Sequential(
            nn.Conv2d(3, 64, 3, bias=False, padding=1, stride=1),  # Increased stride
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        self.conv3X3_1 = Block(64, 64, 3, True)
        self.conv3X3_2 = Block(64, 128, 3)
        self.conv3X3_3 = Block(128, 128, 3, True)
        self.conv3X3_4 = Block(128, 256, 3)
        self.conv3X3_5 = Block(256, 256, 3, True)

        self.conv3X3_gen = Block(256, 500, 3)
        self.conv3X3_age = Block(256, 500, 3)
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = Classifier(256, task)

    def forward(self, x):
        x = checkpoint(self.conv5X5, x)
        x = checkpoint(self.conv3X3_1, x)
        x = checkpoint(self.conv3X3_2, x)
        x = checkpoint(self.conv3X3_3, x)
        x = checkpoint(self.conv3X3_4, x)
        x = checkpoint(self.conv3X3_5, x)

        x = self.avgpool(x)
        x = self.classifier(x)
        
        return x