import torchvision
import torch.nn as nn

class SVCNN(nn.Module):
    def __init__(self, num_classes, architecture_name, feature_dim, pretrained=True):
        super(SVCNN, self).__init__()

        self.num_classes = num_classes
        self.architecture_name = architecture_name
        self.feature_dim = feature_dim

        if self.architecture_name == 'RESNET18':
            self.net = torchvision.models.resnet18(pretrained=pretrained)
 
        self.net.fc = nn.Linear(self.feature_dim, self.num_classes)

    def forward(self, x, feature=False):
        if feature == False:
            return self.net(x)
        else:
            features = nn.Sequential(*list(self.net.children())[:-1])(x)
            outputs = self.net.fc(features.view(x.shape[0], -1))

            return features, outputs

