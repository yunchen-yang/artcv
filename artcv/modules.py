from torchvision.models.resnet import ResNet
import torch
from torch import nn as nn
import collections
import torch.nn.functional as F


class ResNet_CNN(ResNet):
    def __init__(self, block, layers, weight_path, **kwargs):
        super().__init__(block, layers, **kwargs)
        self.weight_path = weight_path
        if self.weight_path is not None:
            self.load_state_dict(torch.load(self.weight_path))
        del self.fc

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

        return x


class Classifier(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden, n_layers, task='ml', use_batch_norm=True, dropout_rate=0.01):
        super().__init__()
        dims = [dim_in] + [dim_hidden]*(n_layers-1) + [dim_out]
        self.task = task
        self.use_batch_norm = use_batch_norm
        self.dropout_rate = dropout_rate
        self.classifier = nn.Sequential(collections.OrderedDict(
            [('Layer {}'.format(i), nn.Sequential(
                nn.Linear(n_in, n_out),
                nn.BatchNorm1d(n_out, momentum=.01, eps=0.001) if self.use_batch_norm else None,
                nn.ReLU() if i != len(dims)-2 else None,
                nn.Dropout(p=self.dropout_rate) if self.dropout_rate > 0 else None))
             for i, (n_in, n_out) in enumerate(zip(dims[:-1], dims[1:]))]))

    def get_logits(self, x):
        for layers in self.classifier:
            for layer in layers:
                if layer is not None:
                    x = layer(x)
        return x

    def forward(self, x):
        if self.task == 'mc':
            return F.softmax(self.get_logits(x), dim=-1)
        elif self.task == 'ml':
            return torch.sigmoid(self.get_logits(x))
        else:
            raise ValueError("The task tag must be either 'ml' (multi-label) or 'mc' (multi-class)!")