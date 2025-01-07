from torchvision import models
import torch.nn as nn
import enum

class Task(enum.Enum):
    GENDER = 1
    AGE = 2
    GENDER_AGE = 3
    RACE = 4
    GENDER_RACE = 5
    AGE_RACE = 6
    ALL = 7

class Classifier(nn.Module):
    def __init__(self, in_features, task):
        super(Classifier, self).__init__()
        self.task = task
        self.fc_gender = nn.Sequential(nn.Linear(in_features, 512),
                                        nn.ReLU(),
                                        nn.Linear(512, 1))  

        self.fc_age = nn.Sequential(nn.Linear(in_features, 1024),    
                                    nn.ReLU(),
                                    nn.Linear(1024, 9))

        self.fc_race = nn.Sequential(nn.Linear(in_features, 1024),
                                     nn.ReLU(),
                                     nn.Linear(1024, 7))
    def forward(self, x):
        if self.task == Task.GENDER:
            return self.fc_gender(x)
        elif self.task == Task.AGE:
            return self.fc_age(x)
        elif self.task == Task.RACE:
            return self.fc_race(x)
        elif self.task == Task.GENDER_AGE:
            return self.fc_gender(x), self.fc_age(x)
        elif self.task == Task.GENDER_RACE:
            return self.fc_gender(x), self.fc_race(x)
        elif self.task == Task.AGE_RACE:
            return self.fc_age(x),self.fc_race(x)
        elif self.task == Task.ALL:
            return self.fc_gender(x), self.fc_age(x), self.fc_race(x)
        

class ResNet18(nn.Module):
    def __init__(self, out_channels, task, freeze=True):
        super(ResNet18, self).__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        in_features = self.model.fc.in_features
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

            self.model.avgpool = nn.Sequential(nn.Conv2d(in_features, out_channels, 3, 1, 1),
                                            nn.BatchNorm2d(out_channels),  
                                            nn.ReLU(),
                                            nn.AdaptiveAvgPool2d(1))
        else:
            out_channels = in_features
            
        self.model.fc = Classifier(out_channels, task)

    def forward(self, x):
        return self.model(x)

class EfficientNetB0(nn.Module):
    def __init__(self, out_channels, task, freeze=True):
        super(EfficientNetB0, self).__init__()
        self.model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
                
            self.model.avgpool = nn.Sequential(nn.Conv2d(1280, out_channels, 3, 1, 1),
                                            nn.BatchNorm2d(out_channels),
                                            nn.ReLU(),
                                            nn.AdaptiveAvgPool2d(1))
        else:
            out_channels = 1280
        self.model.classifier = Classifier(out_channels, task)

    def forward(self, x):
        return self.model(x)