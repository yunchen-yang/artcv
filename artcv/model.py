import numpy as np
import torch
import torch.nn as nn
import torchvision.models.resnet as resnet
import collections
import torch.nn.functional as F
from artcv.modules import ResNet_CNN, Classifier
from artcv.groups_info import _label_groups0, _label_groups1,  _label_groups3, _label_groups4


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
                 use_batch_norm=True, dropout_rate=0.1,
                 weight_path=None, freeze_cnn=False,
                 focal_loss=False, focal_loss_mc=False, alpha_t_mc=True,
                 alpha=(0.25, 0.25, 0.25, 0.25), alpha_mc=(0.25, 0.75, 0.75, 0.75, 0.75, 0.75),
                 gamma_mc=2, gamma=(2, 2, 2, 2), alpha_t=True, alpha_group=(1, 1), gamma_group=2,
                 hierarchical=False, label_groups=(0, 1, 1, 1, 1, 1), group_classifier_kwargs=dict(),
                 weight_group=None,
                 hierarchical_ml=(False, False, False, False),
                 label_groups0=None, label_groups1=None, label_groups3=None, label_groups4=None,
                 group_classifier_kwargs0=dict(), group_classifier_kwargs1=dict(), group_classifier_kwargs3=dict(),
                 group_classifier_kwargs4=dict(), weight_group_ml=(None, None, None, None)):
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
        self.focal_loss_mc = focal_loss_mc
        self.hierarchical = hierarchical
        self.hierarchical_ml = hierarchical_ml

        if self.focal_loss:
            self.alpha = alpha
            self.gamma = gamma
            self.alpha_t = alpha_t
        if self.focal_loss_mc:
            self.alpha_mc =alpha_mc
            self.gamma_mc = gamma_mc
            self.alpha_t_mc = alpha_t_mc
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

        if self.hierarchical_ml[0]:
            self.label_groups0 = np.array(label_groups0) if label_groups0 is not None else np.array(_label_groups0)
            self.num_groups0 = len(np.unique(self.label_groups0))
            self.group_classifier_kwargs0 = {'dim_hidden': self.classifier_hidden[0],
                                             'n_layers': self.classifier_layers[0],
                                             'task': self.task[0],
                                             'use_batch_norm': self.use_batch_norm,
                                             'dropout_rate': self.dropout_rate}
            self.group_classifier_kwargs0.update(group_classifier_kwargs0)
            self.group_classifier0 = Classifier(dim_in=512 * getattr(resnet, BLOCK[tag]).expansion,
                                                dim_out=self.num_groups0, **self.group_classifier_kwargs0)
            self.weight_group0 = weight_group_ml[0] if weight_group_ml[0] is not None else self.weights[0]
            self.group_idx_list0 = torch.nn.ParameterList([torch.nn.Parameter(
                torch.tensor((self.label_groups0 == i).astype(np.uint8), dtype=torch.bool), requires_grad=False)
                for i in range(self.num_groups0)])

        if self.hierarchical_ml[1]:
            self.label_groups1 = np.array(label_groups1) if label_groups1 is not None else np.array(_label_groups1)
            self.num_groups1 = len(np.unique(self.label_groups1))
            self.group_classifier_kwargs1 = {'dim_hidden': self.classifier_hidden[1],
                                             'n_layers': self.classifier_layers[1],
                                             'task': self.task[1],
                                             'use_batch_norm': self.use_batch_norm,
                                             'dropout_rate': self.dropout_rate}
            self.group_classifier_kwargs1.update(group_classifier_kwargs1)
            self.group_classifier1 = Classifier(dim_in=512 * getattr(resnet, BLOCK[tag]).expansion,
                                                dim_out=self.num_groups1, **self.group_classifier_kwargs1)
            self.weight_group1 = weight_group_ml[1] if weight_group_ml[1] is not None else self.weights[1]
            self.group_idx_list1 = torch.nn.ParameterList([torch.nn.Parameter(
                torch.tensor((self.label_groups1 == i).astype(np.uint8), dtype=torch.bool), requires_grad=False)
                for i in range(self.num_groups1)])

        if self.hierarchical_ml[2]:
            self.label_groups3 = np.array(label_groups3) if label_groups3 is not None else np.array(_label_groups3)
            self.num_groups3 = len(np.unique(self.label_groups3))
            self.group_classifier_kwargs3 = {'dim_hidden': self.classifier_hidden[3],
                                             'n_layers': self.classifier_layers[3],
                                             'task': self.task[3],
                                             'use_batch_norm': self.use_batch_norm,
                                             'dropout_rate': self.dropout_rate}
            self.group_classifier_kwargs3.update(group_classifier_kwargs3)
            self.group_classifier3 = Classifier(dim_in=512 * getattr(resnet, BLOCK[tag]).expansion,
                                                dim_out=self.num_groups3, **self.group_classifier_kwargs3)
            self.weight_group3 = weight_group_ml[2] if weight_group_ml[2] is not None else self.weights[3]
            self.group_idx_list3 = torch.nn.ParameterList([torch.nn.Parameter(
                torch.tensor((self.label_groups3 == i).astype(np.uint8), dtype=torch.bool), requires_grad=False)
                for i in range(self.num_groups3)])

        if self.hierarchical_ml[3]:
            self.label_groups4 = np.array(label_groups4) if label_groups4 is not None else np.array(_label_groups4)
            self.num_groups4 = len(np.unique(self.label_groups4))
            self.group_classifier_kwargs4 = {'dim_hidden': self.classifier_hidden[4],
                                             'n_layers': self.classifier_layers[4],
                                             'task': self.task[4],
                                             'use_batch_norm': self.use_batch_norm,
                                             'dropout_rate': self.dropout_rate}
            self.group_classifier_kwargs4.update(group_classifier_kwargs4)
            self.group_classifier4 = Classifier(dim_in=512 * getattr(resnet, BLOCK[tag]).expansion,
                                                dim_out=self.num_groups4, **self.group_classifier_kwargs4)
            self.weight_group4 = weight_group_ml[3] if weight_group_ml[3] is not None else self.weights[4]
            self.group_idx_list4 = torch.nn.ParameterList([torch.nn.Parameter(
                torch.tensor((self.label_groups4 == i).astype(np.uint8), dtype=torch.bool), requires_grad=False)
                for i in range(self.num_groups4)])

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
        if self.hierarchical_ml[0]:
            y_group0 = self.group_classifier0(x_features)
            y_probs0 = torch.zeros_like(y_pred0)
            for i, group_idx in enumerate(self.group_idx_list0):
                y_probs0[:, group_idx] = y_pred0[:, group_idx] / \
                                            (y_pred0[:, group_idx].sum(dim=-1, keepdim=True) + 1e-8) * \
                                            y_group0[:, [i]]
        else:
            y_probs0 = y_pred0.view(-1, self.num_labels[0])
            y_group0 = None

        if self.hierarchical_ml[1]:
            y_group1 = self.group_classifier1(x_features)
            y_probs1 = torch.zeros_like(y_pred1)
            for i, group_idx in enumerate(self.group_idx_list1):
                y_probs1[:, group_idx] = y_pred1[:, group_idx] / \
                                            (y_pred1[:, group_idx].sum(dim=-1, keepdim=True) + 1e-8) * \
                                            y_group1[:, [i]]
        else:
            y_probs1 = y_pred1.view(-1, self.num_labels[1])
            y_group1 = None

        if self.hierarchical_ml[2]:
            y_group3 = self.group_classifier3(x_features)
            y_probs3 = torch.zeros_like(y_pred3)
            for i, group_idx in enumerate(self.group_idx_list3):
                y_probs3[:, group_idx] = y_pred3[:, group_idx] / \
                                            (y_pred3[:, group_idx].sum(dim=-1, keepdim=True) + 1e-8) * \
                                            y_group3[:, [i]]
        else:
            y_probs3 = y_pred3.view(-1, self.num_labels[3])
            y_group3 = None

        if self.hierarchical_ml[3]:
            y_group4 = self.group_classifier4(x_features)
            y_probs4 = torch.zeros_like(y_pred4)
            for i, group_idx in enumerate(self.group_idx_list4):
                y_probs4[:, group_idx] = y_pred4[:, group_idx] / \
                                            (y_pred4[:, group_idx].sum(dim=-1, keepdim=True) + 1e-8) * \
                                            y_group4[:, [i]]
        else:
            y_probs4 = y_pred4.view(-1, self.num_labels[4])
            y_group4 = None

        if self.hierarchical:
            y_probs2, y_group2 = self.get_probs_mc(x_features)
        else:
            y_probs2 = self.get_probs_mc(x_features)
            y_group2 = None

        return y_probs0, y_probs1, y_probs2, y_probs3, y_probs4, (y_group0, y_group1, y_group2, y_group3, y_group4)

    def get_loss_mc(self, x, y2, reduction='sum'):
        if self.hierarchical:
            y_pred2, y_group2 = self.get_probs_mc(self.inference(x))
            if self.focal_loss_mc:
                loss_group = focal_loss_mc(y_group2,
                                           torch.tensor(list(map(lambda x: self.label_groups[x],
                                                                 y2.view(-1)))).view(-1),
                                           num_classes=self.num_groups, alpha=self.alpha_group,
                                           gamma=self.gamma_group, alpha_t=self.alpha_t_mc)
            else:
                loss_group = F.cross_entropy(y_group2,
                                             torch.tensor(list(map(lambda x: self.label_groups[x],
                                                                   y2.view(-1)))).view(-1), reduction='none')
        else:
            y_pred2 = self.get_probs_mc(self.inference(x))
        if self.focal_loss_mc:
            loss2 = focal_loss_mc(y_pred2, y2.view(-1), num_classes=self.num_labels[2],
                                  alpha=self.alpha_mc, gamma=self.gamma_mc, alpha_t=self.alpha_t_mc)
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
        y_pred0, y_pred1, y_pred2, y_pred3, y_pred4, y_groups_tuples = self.get_probs(x)
        y_group0, y_group1, y_group2, y_group3, y_group4 = y_groups_tuples
        if self.focal_loss_mc:
            if self.hierarchical:
                loss_group2 = focal_loss_mc(y_group2,
                                           torch.tensor(list(map(lambda x: self.label_groups[x],
                                                                 y2.view(-1)))).view(-1),
                                           num_classes=self.num_groups, alpha=self.alpha_group,
                                           gamma=self.gamma_group, alpha_t=self.alpha_t_mc)
            else:
                loss_group2 = None
            loss2 = focal_loss_mc(y_pred2, y2.view(-1), num_classes=self.num_labels[2],
                                  alpha=self.alpha_mc, gamma=self.gamma_mc, alpha_t=self.alpha_t_mc)
        else:
            if self.hierarchical:
                loss_group2 = F.cross_entropy(y_group2,
                                             torch.tensor(list(map(lambda x: self.label_groups[x],
                                                                   y2.view(-1)))).view(-1), reduction='none')
            else:
                loss_group2 = None
            loss2 = F.cross_entropy(y_pred2, y2.view(-1), reduction='none')

        if self.focal_loss:
            loss0 = torch.mean(focal_loss_ml(y_pred0, y0, alpha=self.alpha[0], gamma=self.gamma[0],
                                             alpha_t=self.alpha_t), dim=1)
            if self.hierarchical_ml[0]:
                loss_group0 = torch.mean(focal_loss_ml(y_group0,
                                                       torch.cat(list(map(lambda i:
                                                                          torch.sum(
                                                                              F.one_hot(
                                                                                  i, num_classes=self.num_groups0),
                                                                              dim=0).view(1, -1),
                                                                          list(map(lambda x: torch.unique((x * (
                                                                                  torch.tensor(
                                                                                      self.label_groups0)+1)-1),
                                                                                               dim=-1)[1:],
                                                                                   y0.long())))),
                                                                 dim=0).float(),
                                                       alpha=self.alpha[0], gamma=self.gamma[0],
                                                       alpha_t=self.alpha_t), dim=1)
            else:
                loss_group0 = None

            loss1 = torch.mean(focal_loss_ml(y_pred1, y1, alpha=self.alpha[1], gamma=self.gamma[1],
                                             alpha_t=self.alpha_t), dim=1)
            if self.hierarchical_ml[1]:
                loss_group1 = torch.mean(focal_loss_ml(y_group1,
                                                       torch.cat(list(map(lambda i:
                                                                          torch.sum(
                                                                              F.one_hot(
                                                                                  i, num_classes=self.num_groups1),
                                                                              dim=0).view(1, -1),
                                                                          list(map(lambda x: torch.unique((x * (
                                                                                  torch.tensor(
                                                                                      self.label_groups1)+1)-1),
                                                                                               dim=-1)[1:],
                                                                                   y1.long())))),
                                                                 dim=0).float(),
                                                       alpha=self.alpha[1], gamma=self.gamma[1],
                                                       alpha_t=self.alpha_t), dim=1)
            else:
                loss_group1 = None

            loss3 = torch.mean(focal_loss_ml(y_pred3, y3, alpha=self.alpha[2], gamma=self.gamma[2],
                                             alpha_t=self.alpha_t), dim=1)
            if self.hierarchical_ml[2]:
                loss_group3 = torch.mean(focal_loss_ml(y_group3,
                                                       torch.cat(list(map(lambda i:
                                                                          torch.sum(
                                                                              F.one_hot(
                                                                                  i, num_classes=self.num_groups3),
                                                                              dim=0).view(1, -1),
                                                                          list(map(lambda x: torch.unique((x * (
                                                                                  torch.tensor(
                                                                                      self.label_groups3)+1)-1),
                                                                                               dim=-1)[1:],
                                                                                   y3.long())))),
                                                                 dim=0).float(),
                                                       alpha=self.alpha[2], gamma=self.gamma[2],
                                                       alpha_t=self.alpha_t), dim=1)
            else:
                loss_group3 = None

            loss4 = torch.mean(focal_loss_ml(y_pred4, y4, alpha=self.alpha[3], gamma=self.gamma[3],
                                             alpha_t=self.alpha_t), dim=1)
            if self.hierarchical_ml[3]:
                loss_group4 = torch.mean(focal_loss_ml(y_group4,
                                                       torch.cat(list(map(lambda i:
                                                                          torch.sum(
                                                                              F.one_hot(
                                                                                  i, num_classes=self.num_groups4),
                                                                              dim=0).view(1, -1),
                                                                          list(map(lambda x: torch.unique((x * (
                                                                                  torch.tensor(
                                                                                      self.label_groups4)+1)-1),
                                                                                               dim=-1)[1:],
                                                                                   y4.long())))),
                                                                 dim=0).float(),
                                                       alpha=self.alpha[3], gamma=self.gamma[3],
                                                       alpha_t=self.alpha_t), dim=1)
            else:
                loss_group4 = None

        else:
            loss0 = torch.mean(F.binary_cross_entropy(y_pred0, y0, reduction='none'), dim=1)
            if self.hierarchical_ml[0]:
                loss_group0 = torch.mean(F.binary_cross_entropy(y_group0,
                                                                torch.cat(list(map(lambda i:
                                                                                   torch.sum(
                                                                                      F.one_hot(i,
                                                                                                num_classes=
                                                                                                self.num_groups0),
                                                                                      dim=0).view(1, -1),
                                                                                   list(map(lambda x: torch.unique((
                                                                                           x * (
                                                                                               torch.tensor(
                                                                                                   self.label_groups0)
                                                                                               + 1) - 1),
                                                                                                       dim=-1)[1:],
                                                                                            y0.long()))
                                                                                   )), dim=0).float(),
                                                                reduction='none'), dim=1)
            else:
                loss_group0 = None

            loss1 = torch.mean(F.binary_cross_entropy(y_pred1, y1, reduction='none'), dim=1)
            if self.hierarchical_ml[1]:
                loss_group1 = torch.mean(F.binary_cross_entropy(y_group1,
                                                                torch.cat(list(map(lambda i:
                                                                                   torch.sum(
                                                                                      F.one_hot(i,
                                                                                                num_classes=
                                                                                                self.num_groups1),
                                                                                      dim=0).view(1, -1),
                                                                                   list(map(lambda x: torch.unique((
                                                                                           x * (
                                                                                               torch.tensor(
                                                                                                   self.label_groups1)
                                                                                               + 1) - 1),
                                                                                                       dim=-1)[1:],
                                                                                            y1.long()))
                                                                                   )), dim=0).float(),
                                                                reduction='none'), dim=1)
            else:
                loss_group1 = None

            loss3 = torch.mean(F.binary_cross_entropy(y_pred3, y3, reduction='none'), dim=1)
            if self.hierarchical_ml[2]:
                loss_group3 = torch.mean(F.binary_cross_entropy(y_group3,
                                                                torch.cat(list(map(lambda i:
                                                                                   torch.sum(
                                                                                      F.one_hot(i,
                                                                                                num_classes=
                                                                                                self.num_groups3),
                                                                                      dim=0).view(1, -1),
                                                                                   list(map(lambda x: torch.unique((
                                                                                           x * (
                                                                                               torch.tensor(
                                                                                                   self.label_groups3)
                                                                                               + 1) - 1),
                                                                                                       dim=-1)[1:],
                                                                                            y3.long()))
                                                                                   )), dim=0).float(),
                                                                reduction='none'), dim=1)
            else:
                loss_group3 = None

            loss4 = torch.mean(F.binary_cross_entropy(y_pred4, y4, reduction='none'), dim=1)
            if self.hierarchical_ml[3]:
                loss_group4 = torch.mean(F.binary_cross_entropy(y_group4,
                                                                torch.cat(list(map(lambda i:
                                                                                   torch.sum(
                                                                                      F.one_hot(i,
                                                                                                num_classes=
                                                                                                self.num_groups4),
                                                                                      dim=0).view(1, -1),
                                                                                   list(map(lambda x: torch.unique((
                                                                                           x * (
                                                                                               torch.tensor(
                                                                                                   self.label_groups4)
                                                                                               + 1) - 1),
                                                                                                       dim=-1)[1:],
                                                                                            y4.long()))
                                                                                   )), dim=0).float(),
                                                                reduction='none'), dim=1)
            else:
                loss_group4 = None

        return loss0, loss1, loss2, loss3, loss4, (loss_group0, loss_group1, loss_group2, loss_group3, loss_group4)

    def forward(self, x, y0, y1, y2, y3, y4, reduction='sum'):
        loss0, loss1, loss2, loss3, loss4, loss_groups_tuple = self.get_loss(x, y0, y1, y2, y3, y4)
        loss_group0, loss_group1, loss_group2, loss_group3, loss_group4 = loss_groups_tuple

        loss_groups = 0
        if loss_group0 is not None:
            loss_groups += loss_group0 * self.weight_group0

        if loss_group1 is not None:
            loss_groups += loss_group1 * self.weight_group1

        if loss_group3 is not None:
            loss_groups += loss_group3 * self.weight_group3

        if loss_group4 is not None:
            loss_groups += loss_group4 * self.weight_group4

        if loss_group2 is not None:
            loss_groups += loss_group2 * self.weight_group

        if reduction == 'sum':
            return loss0 * self.weights[0] + loss1 * self.weights[1] + loss2 * self.weights[2] + \
                   loss3 * self.weights[3] + loss4 * self.weights[4] + loss_groups
        elif reduction == 'none':
            return loss0 * self.weights[0], loss1 * self.weights[1], loss2 * self.weights[2], \
                   loss3 * self.weights[3], loss4 * self.weights[4], loss_groups

    def get_concat_probs(self, x, return_hier_pred=False):
        if return_hier_pred:
            y_pred0, y_pred1, y_pred2, y_pred3, y_pred4, y_groups_tuples = self.get_probs(x)
            _, _, y_group2, _, _ = y_groups_tuples
            return torch.cat((y_pred0, y_pred1,
                              F.one_hot(y_pred2.argmax(axis=-1), num_classes=self.num_labels[2]).float()[:, 1:],
                              y_pred3, y_pred4), dim=1), \
                   F.one_hot(y_group2.argmax(axis=-1), num_classes=self.num_groups)
        else:
            y_pred0, y_pred1, y_pred2, y_pred3, y_pred4, _ = self.get_probs(x)

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