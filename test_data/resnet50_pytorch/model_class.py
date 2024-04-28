import torch
import torchvision.models as models

class ResNet50(torch.nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        res_net_50 = models.resnet50()


        self.conv1 = res_net_50.conv1
        self.bn1 = res_net_50.bn1
        self.relu = res_net_50.relu
        self.maxpool = res_net_50.maxpool
        self.layer1 = res_net_50.layer1
        self.layer2 = res_net_50.layer2
        self.layer3 = res_net_50.layer3
        self.layer4 = res_net_50.layer4
        self.avgpool = res_net_50.avgpool
        self.fc = res_net_50.fc

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

