import numpy as np
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
                 weight_path=None, freeze_cnn=False,
                 focal_loss=False, alpha=(0.25, 0.25, 0.25, 0.25), alpha_mc=(0.25, 0.75, 0.75, 0.75, 0.75, 0.75),
                 gamma_mc=2, gamma=(2, 2, 2, 2), alpha_t=True, alpha_group=(1, 1), gamma_group=2,
                 hierarchical=False, label_groups=(0, 1, 1, 1, 1, 1), group_classifier_kwargs=dict(),
                 weight_group=None):
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
        self.focal_loss = focal_loss
        self.hierarchical = hierarchical

        if self.focal_loss:
            self.alpha = alpha
            self.alpha_mc =alpha_mc
            self.gamma = gamma
            self.gamma_mc = gamma_mc
            self.alpha_t = alpha_t
            if self.hierarchical:
                self.alpha_group = alpha_group
                self.gamma_group = gamma_group

        self.cnn = ResNet_CNN(getattr(resnet, BLOCK[tag]), LAYERS[tag],
                              weight_path=self.weight_path, freeze_layers=self.freeze_cnn)

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

        if self.hierarchical:
            self.label_groups = np.array(label_groups)
            self.num_groups = len(np.unique(self.label_groups))
            self.group_classifier_kwargs = {'dim_hidden': self.classifier_hidden[2],
                                            'n_layers': self.classifier_layers[2],
                                            'task': self.task[2],
                                            'use_batch_norm': self.use_batch_norm,
                                            'dropout_rate': self.dropout_rate}
            self.group_classifier_kwargs.update(group_classifier_kwargs)
            self.group_classifier = Classifier(dim_in=512 * getattr(resnet, BLOCK[tag]).expansion,
                                               dim_out=self.num_groups, **self.group_classifier_kwargs)
            self.weight_group = weight_group if weight_group is not None else self.weights[2]
            self.group_idx_list = torch.nn.ParameterList([torch.nn.Parameter(
                torch.tensor((self.label_groups == i).astype(np.uint8), dtype=torch.bool), requires_grad=False)
                for i in range(self.num_groups)])

    def inference(self, x):
        return self.cnn(x)

    def get_probs_mc(self, x_features):
        y_pred2 = self.classifiers['classifier2'](x_features)

        if self.hierarchical:
            y_group2 = self.group_classifier(x_features)
            y_weighted2 = torch.zeros_like(y_pred2)
            for i, group_idx in enumerate(self.group_idx_list):
                y_weighted2[:, group_idx] = y_pred2[:, group_idx] / \
                                            (y_pred2[:, group_idx].sum(dim=-1, keepdim=True) + 1e-8) * \
                                            y_group2[:, [i]]
            return y_weighted2.view(-1, self.num_labels[2]), y_group2.view(-1, self.num_groups)
        else:
            return y_pred2.view(-1, self.num_labels[2])

    def get_probs(self, x):
        x_features = self.inference(x)
        y_pred0 = self.classifiers['classifier0'](x_features)
        y_pred1 = self.classifiers['classifier1'](x_features)
        y_pred3 = self.classifiers['classifier3'](x_features)
        y_pred4 = self.classifiers['classifier4'](x_features)
        if self.hierarchical:
            y_weighted2, y_group2 = self.get_probs_mc(x_features)
            return y_pred0.view(-1, self.num_labels[0]), y_pred1.view(-1, self.num_labels[1]), \
                   y_weighted2, \
                   y_pred3.view(-1, self.num_labels[3]), y_pred4.view(-1, self.num_labels[4]), \
                   y_group2
        else:
            return y_pred0.view(-1, self.num_labels[0]), y_pred1.view(-1, self.num_labels[1]), \
                   self.classifiers['classifier2'](x_features).view(-1, self.num_labels[2]), \
                   y_pred3.view(-1, self.num_labels[3]), y_pred4.view(-1, self.num_labels[4])

    def get_loss_mc(self, x, y2, reduction='sum'):
        if self.hierarchical:
            y_pred2, y_group2 = self.get_probs_mc(self.inference(x))
            if self.focal_loss:
                loss_group = focal_loss_mc(y_group2,
                                           torch.tensor(list(map(lambda x: self.label_groups[x],
                                                                 y2.view(-1)))).view(-1),
                                           num_classes=self.num_groups, alpha=self.alpha_group,
                                           gamma=self.gamma_group, alpha_t=self.alpha_t)
            else:
                loss_group = F.cross_entropy(y_group2,
                                             torch.tensor(list(map(lambda x: self.label_groups[x],
                                                                   y2.view(-1)))).view(-1), reduction='none')
        else:
            y_pred2 = self.get_probs_mc(self.inference(x))
        if self.focal_loss:
            loss2 = focal_loss_mc(y_pred2, y2.view(-1), num_classes=self.num_labels[2],
                                  alpha=self.alpha_mc, gamma=self.gamma_mc, alpha_t=self.alpha_t)
        else:
            loss2 = F.cross_entropy(y_pred2, y2.view(-1), reduction='none')

        if self.hierarchical:
            if reduction == 'sum':
                return loss2 * self.weights[2] + loss_group * self.weight_group
            elif reduction == 'none':
                return loss2 * self.weights[2], loss_group * self.weight_group
        else:
            return loss2 * self.weights[2]

    def get_loss(self, x, y0, y1, y2, y3, y4):
        if self.hierarchical:
            y_pred0, y_pred1, y_pred2, y_pred3, y_pred4, y_group2 = self.get_probs(x)
            if self.focal_loss:
                loss_group = focal_loss_mc(y_group2,
                                           torch.tensor(list(map(lambda x: self.label_groups[x],
                                                                 y2.view(-1)))).view(-1),
                                           num_classes=self.num_groups, alpha=self.alpha_group,
                                           gamma=self.gamma_group, alpha_t=self.alpha_t)
            else:
                loss_group = F.cross_entropy(y_group2,
                                             torch.tensor(list(map(lambda x: self.label_groups[x],
                                                                   y2.view(-1)))).view(-1), reduction='none')
        else:
            y_pred0, y_pred1, y_pred2, y_pred3, y_pred4 = self.get_probs(x)
        if self.focal_loss:
            loss0 = torch.mean(focal_loss_ml(y_pred0, y0, alpha=self.alpha[0], gamma=self.gamma[0],
                                             alpha_t=self.alpha_t), dim=1)
            loss1 = torch.mean(focal_loss_ml(y_pred1, y1, alpha=self.alpha[1], gamma=self.gamma[1],
                                             alpha_t=self.alpha_t), dim=1)
            loss3 = torch.mean(focal_loss_ml(y_pred3, y3, alpha=self.alpha[2], gamma=self.gamma[2],
                                             alpha_t=self.alpha_t), dim=1)
            loss4 = torch.mean(focal_loss_ml(y_pred4, y4, alpha=self.alpha[3], gamma=self.gamma[3],
                                             alpha_t=self.alpha_t), dim=1)
            loss2 = focal_loss_mc(y_pred2, y2.view(-1), num_classes=self.num_labels[2],
                                  alpha=self.alpha_mc, gamma=self.gamma_mc, alpha_t=self.alpha_t)

        else:
            loss0 = torch.mean(F.binary_cross_entropy(y_pred0, y0, reduction='none'), dim=1)
            loss1 = torch.mean(F.binary_cross_entropy(y_pred1, y1, reduction='none'), dim=1)
            loss3 = torch.mean(F.binary_cross_entropy(y_pred3, y3, reduction='none'), dim=1)
            loss4 = torch.mean(F.binary_cross_entropy(y_pred4, y4, reduction='none'), dim=1)
            loss2 = F.cross_entropy(y_pred2, y2.view(-1), reduction='none')
        if self.hierarchical:
            return loss0, loss1, loss2, loss3, loss4, loss_group
        else:
            return loss0, loss1, loss2, loss3, loss4

    def forward(self, x, y0, y1, y2, y3, y4, reduction='sum'):
        if self.hierarchical:
            loss0, loss1, loss2, loss3, loss4, loss_group = self.get_loss(x, y0, y1, y2, y3, y4)
            if reduction == 'sum':
                return loss0 * self.weights[0] + loss1 * self.weights[1] + loss2 * self.weights[2] + \
                       loss3 * self.weights[3] + loss4 * self.weights[4] + loss_group * self.weight_group
            elif reduction == 'none':
                return loss0 * self.weights[0], loss1 * self.weights[1], loss2 * self.weights[2], \
                       loss3 * self.weights[3], loss4 * self.weights[4], loss_group * self.weight_group
        else:
            loss0, loss1, loss2, loss3, loss4 = self.get_loss(x, y0, y1, y2, y3, y4)
            if reduction == 'sum':
                return loss0*self.weights[0] + loss1*self.weights[1] + loss2*self.weights[2] + \
                       loss3*self.weights[3] + loss4*self.weights[4]
            elif reduction == 'none':
                return loss0*self.weights[0], loss1*self.weights[1], loss2*self.weights[2], \
                       loss3*self.weights[3], loss4*self.weights[4]

    def get_concat_probs(self, x, return_hier_pred=False):
        if self.hierarchical:
            if return_hier_pred:
                y_pred0, y_pred1, y_pred2, y_pred3, y_pred4, y_group2 = self.get_probs(x)
                return torch.cat((y_pred0, y_pred1,
                                  F.one_hot(y_pred2.argmax(axis=-1), num_classes=self.num_labels[2]).float()[:, 1:],
                                  y_pred3, y_pred4), dim=1), \
                       F.one_hot(y_group2.argmax(axis=-1), num_classes=self.num_groups)
            else:
                y_pred0, y_pred1, y_pred2, y_pred3, y_pred4, _ = self.get_probs(x)
        else:
            y_pred0, y_pred1, y_pred2, y_pred3, y_pred4 = self.get_probs(x)
        return torch.cat((y_pred0, y_pred1,
                          F.one_hot(y_pred2.argmax(axis=-1), num_classes=self.num_labels[2]).float()[:, 1:],
                          y_pred3, y_pred4), dim=1)


def focal_loss_ml(inputs, targets, alpha=0.25, gamma=2, alpha_t=False):
    BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
    pt = torch.exp(-BCE_loss)
    if alpha_t:
        return (-alpha * (targets * 2 -1) + targets) * (1-pt)**gamma * BCE_loss
    else:
        return alpha * (1-pt)**gamma * BCE_loss


def focal_loss_mc(inputs, targets,
                  num_classes, alpha=(0.25, 0.75, 0.75, 0.75, 0.75, 0.75), gamma=2, alpha_t=False):
    targets_one_hot = F.one_hot(targets, num_classes=num_classes)
    pt = inputs * targets_one_hot
    one_sub_pt = 1 - pt
    log_pt = targets_one_hot * torch.log(inputs + 1e-6)
    if alpha_t:
        return torch.sum((-torch.tensor(alpha))*one_sub_pt**gamma*log_pt, dim=-1)
    else:
        return torch.sum((-1)*one_sub_pt**gamma*log_pt, dim=-1)


def get_weight_mat(y, ratio=10, base=10):
    return (torch.ones(y.shape) - y) * (torch.sum(y, dim=-1).view(-1, 1) + base) \
           * ratio / (y.shape[1] - torch.sum(y, dim=-1).view(-1, 1)).detach()