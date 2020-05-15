import torch
import torch.nn as nn
import torchvision.models.resnet as resnet
import collections
import torch.nn.functional as F
from artcv.modules import ResNet_CNN, Classifier


BLOCK = {'18': 'BasicBlock',
         '34': 'BasicBlock',
         '50': 'Bottleneck',
         '101': 'Bottleneck',
         '152': 'Bottleneck'}


LAYERS = {'18': [2, 2, 2, 2],
          '34': [3, 4, 6, 3],
          '50': [3, 4, 6, 3],
          '101': [3, 4, 23, 3],
          '152': [3, 8, 36, 3]}


class ArtCV(nn.Module):
    def __init__(self, tag='18', num_labels=(100, 681, 6, 1920, 768),
                 classifier_layers=(1, 1, 1, 1, 1), classifier_hidden=(2048, 2048, 2048, 2048, 2048),
                 task=('ml', 'ml', 'mc', 'ml', 'ml'), weights=(1, 1, 1, 1, 1),
                 use_batch_norm=True, dropout_rate=0.01,
                 weight_path=None, freeze_cnn=False):
        super().__init__()
        self.tag = tag
        self.num_labels = num_labels
        self.classifier_layers = classifier_layers
        self.classifier_hidden = classifier_hidden
        self.task = task
        self.weights = weights
        self.use_batch_norm = use_batch_norm
        self.dropout_rate = dropout_rate
        self.weight_path = weight_path
        self.freeze_cnn = freeze_cnn

        self.cnn = ResNet_CNN(getattr(resnet, BLOCK[tag]), LAYERS[tag],
                              weight_path=self.weight_path, freeze_layers=freeze_cnn)

        self.classifiers = nn.ModuleDict(
            collections.OrderedDict(
                [('classifier{}'.format(i), Classifier(dim_in=512 * getattr(resnet, BLOCK[tag]).expansion,
                                                       dim_out=self.num_labels[i],
                                                       dim_hidden=self.classifier_hidden[i],
                                                       n_layers=self.classifier_layers[i],
                                                       task=self.task[i],
                                                       use_batch_norm=self.use_batch_norm,
                                                       dropout_rate=self.dropout_rate))
                 for i in range(len(self.num_labels))]))

    def inference(self, x):
        return self.cnn(x)

    def get_probs(self, x):
        x_features = self.inference(x)
        y_pred0 = self.classifiers['classifier0'](x_features)
        y_pred1 = self.classifiers['classifier1'](x_features)
        y_pred2 = self.classifiers['classifier2'](x_features)
        y_pred3 = self.classifiers['classifier3'](x_features)
        y_pred4 = self.classifiers['classifier4'](x_features)
        return y_pred0.view(-1, self.num_labels[0]), y_pred1.view(-1, self.num_labels[1]), \
               y_pred2.view(-1, self.num_labels[2]), \
               y_pred3.view(-1, self.num_labels[3]), y_pred4.view(-1, self.num_labels[4])

    def get_loss(self, x, y0, y1, y2, y3, y4):
        y_pred0, y_pred1, y_pred2, y_pred3, y_pred4 = self.get_probs(x)
        loss0 = torch.mean(F.binary_cross_entropy(y_pred0, y0, reduction='none'), dim=1)
        loss1 = torch.mean(F.binary_cross_entropy(y_pred1, y1, reduction='none'), dim=1)
        loss3 = torch.mean(F.binary_cross_entropy(y_pred3, y3, reduction='none'), dim=1)
        loss4 = torch.mean(F.binary_cross_entropy(y_pred4, y4, reduction='none'), dim=1)
        loss2 = F.cross_entropy(y_pred2, y2.view(-1), reduction='none')
        return loss0, loss1, loss2, loss3, loss4

    def forward(self, x, y0, y1, y2, y3, y4):
        loss0, loss1, loss2, loss3, loss4 = self.get_loss(x, y0, y1, y2, y3, y4)
        loss = loss0*self.weights[0] + loss1*self.weights[1] + loss2*self.weights[2] + \
               loss3*self.weights[3] + loss4*self.weights[4]
        return loss

    def get_concat_probs(self, x):
        y_pred0, y_pred1, y_pred2, y_pred3, y_pred4 = self.get_probs(x)

        return torch.cat((y_pred0, y_pred1,
                          F.one_hot(y_pred2.argmax(axis=-1), num_classes=self.num_labels[2]).float()[:, 1:],
                          y_pred3, y_pred4), dim=1)
