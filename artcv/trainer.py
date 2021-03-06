import torch
import sys
import copy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import fbeta_score

import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from tqdm import trange
from artcv.utils import regularized_pred


class Trainer:
    def __init__(self, model, dataset, use_cuda=True,
                 shuffle=True, epochs=100, extra_epochs_mc=1, train_mc_step=1, 
                 batch_size_train=64, batch_size_val=64, batch_size_all=64, batch_size_test=64,
                 head_log=True,
                 monitor_frequency=False, compute_acc=True, printout=False,
                 dataloader_train_kwargs=dict(), dataloader_val_kwargs=dict(),
                 dataloader_all_kwargs=dict(),
                 dataloader_test_kwargs=dict()):
        self.model = model
        self.dataset = dataset
        self.train_val = bool(self.dataset.train_val_split)
        self.use_cuda = use_cuda
        if self.use_cuda and torch.cuda.is_available():
            torch.set_default_tensor_type(torch.cuda.FloatTensor)
        self.epochs = epochs
        self.extra_epochs_mc = extra_epochs_mc
        self.train_mc_step = train_mc_step
        self.head_log = head_log
        self.running_loss = []
        if self.head_log:
            self.loss_log = dict()
            self.loss_log['Head 0'] = []
            self.loss_log['Head 1'] = []
            self.loss_log['Head 2'] = []
            self.loss_log['Head 3'] = []
            self.loss_log['Head 4'] = []
            if self.model.hierarchical:
                self.loss_log['Head group classifiers'] = []

        if self.train_val:
            if shuffle:
                train_sampler = RandomSampler(dataset.train)
                val_sampler = RandomSampler(dataset.val)
            else:
                train_sampler = SequentialSampler(dataset.train)
                val_sampler = SequentialSampler(dataset.val)
            self.dataloader_train_kwargs = copy.deepcopy(dataloader_train_kwargs)
            self.dataloader_train_kwargs.update({'batch_size': batch_size_train, 'sampler': train_sampler})
            self.dataloader_train = DataLoader(self.dataset.train, **self.dataloader_train_kwargs)
            self.dataloader_val_kwargs = copy.deepcopy(dataloader_val_kwargs)
            self.dataloader_val_kwargs.update({'batch_size': batch_size_val, 'sampler': val_sampler})
            self.dataloader_val = DataLoader(self.dataset.val, **self.dataloader_val_kwargs)
            self.loss_history_train = []
            self.loss_history_val = []
            self.accuracy_history_train = []
            self.accuracy_history_val = []

        else:
            sampler = RandomSampler(dataset.all)
            self.dataloader_train_kwargs = copy.deepcopy(dataloader_train_kwargs)
            self.dataloader_train_kwargs.update({'batch_size': batch_size_train, 'sampler': sampler})
            self.dataloader_train = DataLoader(self.dataset.all, **self.dataloader_train_kwargs)
            self.loss_history_train = []
            self.accuracy_history_train = []

        all_sampler = SequentialSampler(dataset.all)
        self.dataloader_all_kwargs = copy.deepcopy(dataloader_all_kwargs)
        self.dataloader_all_kwargs.update({'batch_size': batch_size_all, 'sampler': all_sampler})
        self.dataloader_all = DataLoader(self.dataset.all, **self.dataloader_all_kwargs)

        if self.dataset.test_path is not None:
            test_sampler = SequentialSampler(dataset.test)
            self.dataloader_test_kwargs = copy.deepcopy(dataloader_test_kwargs)
            self.dataloader_test_kwargs.update({'batch_size': batch_size_test, 'sampler': test_sampler})
            self.dataloader_test = DataLoader(self.dataset.test, **self.dataloader_test_kwargs)

        self.use_cuda = use_cuda and torch.cuda.is_available()
        if self.use_cuda:
            self.model.cuda()
        self.monitor_frequency = monitor_frequency
        self.compute_acc = compute_acc
        self.printout = printout

    def before_iter(self):
        pass

    def after_iter(self):
        pass

    def train_mc(self, lr, parameters=None, betas=(0.9, 0.999), eps=1e-8, weight_decay=0,
              grad_clip=False, max_norm=1e-5):
        epochs = self.extra_epochs_mc
        self.model.classifiers['classifier2'].train()
        if self.model.hierarchical:
            self.model.group_classifier.train()
        params = filter(lambda x: x.requires_grad, self.model.parameters()) \
            if parameters is None else parameters
        optim = torch.optim.Adam(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        for epoch in range(epochs):
            for x, _, _, y2, _, _ in self.dataloader_train:
                if self.use_cuda and torch.cuda.is_available():
                    x = x.cuda()
                    y2 = y2.cuda()
                loss = torch.mean(self.model.get_loss_mc(x, y2))
                optim.zero_grad()
                loss.backward()
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(params, max_norm=max_norm)
                optim.step()
                
    def train(self, parameters=None, lr=1e-1, mc_lr=1e-1, betas=(0.9, 0.999), eps=1e-8, weight_decay=0,
              reduce_lr=False, step=5, gamma=0.8,
              reduce_lr_mc=False, step_mc=5, gamma_mc=0.8,
              grad_clip=False, max_norm=1e-5,
              train_mc_kwargs=dict()):
        epochs = self.epochs
        self.model.train()
        params = filter(lambda x: x.requires_grad, self.model.parameters()) \
            if parameters is None else parameters
        optim = torch.optim.Adam(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        if bool(self.extra_epochs_mc):
            epoch_count_mc = 0
        with trange(epochs, desc='Training progress: ', file=sys.stdout) as progbar:
            for epoch_idx in progbar:
                self.before_iter()
                progbar.update(1)
                running_loss = 0
                if self.head_log:
                    head_loss0 = 0
                    head_loss1 = 0
                    head_loss2 = 0
                    head_loss3 = 0
                    head_loss4 = 0
                    if self.model.hierarchical:
                        head_loss_group = 0
                for data_tensors in self.dataloader_train:
                    data_tensor_tuples = [data_tensors]
                    if self.head_log:
                        if self.model.hierarchical:
                            loss0, loss1, loss2, loss3, loss4, loss_group = self.loss(*data_tensor_tuples,
                                                                                      head_log=self.head_log)
                            loss = torch.mean(loss0 + loss1 + loss2 + loss3 + loss4 + loss_group)
                            head_loss_group += torch.mean(loss_group).item()
                        else:
                            loss0, loss1, loss2, loss3, loss4, _ = self.loss(*data_tensor_tuples,
                                                                             head_log=self.head_log)
                            loss = torch.mean(loss0 + loss1 + loss2 + loss3 + loss4)
                        head_loss0 += torch.mean(loss0).item()
                        head_loss1 += torch.mean(loss1).item()
                        head_loss2 += torch.mean(loss2).item()
                        head_loss3 += torch.mean(loss3).item()
                        head_loss4 += torch.mean(loss4).item()
                    else:
                        loss = self.loss(*data_tensor_tuples)
                    running_loss += loss.item()
                    optim.zero_grad()
                    loss.backward()
                    if grad_clip:
                        torch.nn.utils.clip_grad_norm_(params, max_norm=max_norm)
                    optim.step()
                self.running_loss.append(running_loss/len(self.dataloader_train))
                if self.head_log:
                    self.loss_log['Head 0'].append(head_loss0/len(self.dataloader_train))
                    self.loss_log['Head 1'].append(head_loss1/len(self.dataloader_train))
                    self.loss_log['Head 2'].append(head_loss2/len(self.dataloader_train))
                    self.loss_log['Head 3'].append(head_loss3/len(self.dataloader_train))
                    self.loss_log['Head 4'].append(head_loss4/len(self.dataloader_train))
                    if self.model.hierarchical:
                        self.loss_log['Head group classifiers'].append(head_loss_group / len(self.dataloader_train))

                if (epoch_idx+1) % step == 0 & reduce_lr:
                    for p in optim.param_groups:
                        p['lr'] *= gamma

                if bool(self.monitor_frequency):
                    if (epoch_idx + 1) % self.monitor_frequency == 0:
                        current_loss_train = self.compute_loss(tag='train')
                        self.loss_history_train.append(current_loss_train)
                        if self.compute_acc:
                            current_accuracy_train = self.compute_accuracy(tag='train')
                            self.accuracy_history_train.append(current_accuracy_train)
                        if self.train_val:
                            current_loss_val = self.compute_loss(tag='val')
                            self.loss_history_val.append(current_loss_val)
                            if self.compute_acc:
                                current_accuracy_val = self.compute_accuracy(tag='val')
                                self.accuracy_history_val.append(current_accuracy_val)
                        if self.printout:
                            print("After %i epochs, loss is %f and prediction accuracy is %f."
                                  % (epoch_idx, current_loss_train, current_accuracy_train))
                if bool(self.extra_epochs_mc) & ((epoch_idx + 1) % self.train_mc_step == 0):
                    self.train_mc(lr=mc_lr, **train_mc_kwargs)
                    epoch_count_mc += 1
                    if epoch_count_mc % step_mc == 0 & reduce_lr_mc:
                        mc_lr *= gamma_mc
                                  
                self.after_iter()

    def loss(self, data_tensors, head_log=False):
        x, y0, y1, y2, y3, y4 = data_tensors
        if self.use_cuda and torch.cuda.is_available():
            x = x.cuda()
            y0 = y0.cuda()
            y1 = y1.cuda()
            y2 = y2.cuda()
            y3 = y3.cuda()
            y4 = y4.cuda()
        if head_log:
            loss0, loss1, loss2, loss3, loss4, loss_group = self.model(x, y0, y1, y2, y3, y4, reduction='none')
            return loss0, loss1, loss2, loss3, loss4, loss_group

        else:
            loss = torch.mean(self.model(x, y0, y1, y2, y3, y4, reduction='sum'))
            return loss

    @torch.no_grad()
    def plot_running_loss(self, epochs_override=None):
        len_ticks = len(self.running_loss)
        if epochs_override is None:
            x_axis = np.linspace(1, self.epochs, len_ticks)
        else:
            x_axis = np.linspace(1, epochs_override, len_ticks)
        plt.figure()
        plt.plot(x_axis, self.running_loss)
        plt.xlabel('Number of epochs')
        plt.ylabel('Estimated loss')
        plt.show()

    @torch.no_grad()
    def plot_head_loss(self, epochs_override=None):
        if not self.head_log:
            pass
        else:
            len_ticks = len(self.running_loss)
            if epochs_override is None:
                x_axis = np.linspace(1, self.epochs, len_ticks)
            else:
                x_axis = np.linspace(1, epochs_override, len_ticks)
            plt.figure()
            for key in self.loss_log.keys():
                log_array = np.array(self.loss_log[key])
                plt.plot(x_axis, log_array/log_array[0], label=key)
            plt.legend(loc='lower left', bbox_to_anchor=(1, 0.25, 0.5, 0.5))
            plt.xlabel('Number of epochs')
            plt.ylabel('Estimated loss')
            plt.show()

    @torch.no_grad()
    def compute_loss(self, tag):
        self.model.eval()
        loss_sum = 0
        if tag == 'train':
            _dataloader = self.dataloader_train
        elif tag == 'val':
            _dataloader = self.dataloader_val
        elif tag == 'all':
            _dataloader = self.dataloader_all
        else:
            raise ValueError('Invalid tag!')
        _dataset = _dataloader.dataset
        for data_tensors in _dataloader:
            x, y0, y1, y2, y3, y4 = data_tensors
            if self.use_cuda and torch.cuda.is_available():
                x = x.cuda()
                y0 = y0.cuda()
                y1 = y1.cuda()
                y2 = y2.cuda()
                y3 = y3.cuda()
                y4 = y4.cuda()
            loss = self.model(x, y0, y1, y2, y3, y4)
            loss_sum += torch.sum(loss).item()
        loss_mean = loss_sum / len(_dataset)
        self.model.train()
        return loss_mean

    @torch.no_grad()
    def loss_history_plot(self, epochs_override=None):
        len_ticks = len(self.loss_history_train)
        if epochs_override is None:
            x_axis = np.linspace(1, self.epochs, len_ticks)
        else:
            x_axis = np.linspace(1, epochs_override, len_ticks)
        plt.figure()
        if self.train_val:
            assert (len(self.loss_history_train) == len(self.loss_history_val))
            plt.plot(x_axis, self.loss_history_train, label='Training set')
            plt.plot(x_axis, self.loss_history_val, label='Validation set')
            plt.legend()
        else:
            plt.plot(x_axis, self.loss_history_train)

        plt.xlabel('Number of epochs')
        plt.ylabel('Loss')
        plt.show()

    @torch.no_grad()
    def get_probs(self, tag, return_hier_pred=False, return_probs_only=False):
        self.model.eval()
        predictions_tem = []
        if return_hier_pred:
            hier_pred_tem = []
        if tag == 'test':
            _dataloader = self.dataloader_test
            for data_tensors in _dataloader:
                x = data_tensors
                if self.use_cuda and torch.cuda.is_available():
                    x = x.cuda()
                y_concat_prob = self.model.get_concat_probs(x)
                predictions_tem += [y_concat_prob]
            predictions_array = torch.cat(predictions_tem).detach().cpu().numpy()
            return predictions_array
        else:
            ground_truth = []
            if tag == 'train':
                _dataloader = self.dataloader_train
            elif tag == 'val':
                _dataloader = self.dataloader_val
            elif tag == 'all':
                _dataloader = self.dataloader_all
            else:
                raise ValueError('Invalid tag!')
            for data_tensors in _dataloader:
                if not return_probs_only:
                    x, y0, y1, y2, y3, y4 = data_tensors
                    if self.use_cuda and torch.cuda.is_available():
                        x = x.cuda()
                        y0 = y0.cuda()
                        y1 = y1.cuda()
                        y2 = y2.cuda()
                        y3 = y3.cuda()
                        y4 = y4.cuda()
                    ground_truth += [torch.cat((y0.long(),
                                                y1.long(),
                                                F.one_hot(y2, num_classes=6).squeeze()[:, 1:].long(),
                                                y3.long(),
                                                y4.long()), dim=1)]
                else:
                    x, _, _, _, _, _ = data_tensors
                    if self.use_cuda and torch.cuda.is_available():
                        x = x.cuda()
                if return_hier_pred:
                    y_concat_prob, y_group_pred = self.model.get_concat_probs(x, return_hier_pred=return_hier_pred)
                    hier_pred_tem += [y_group_pred]
                else:
                    y_concat_prob = self.model.get_concat_probs(x, return_hier_pred=return_hier_pred)
                predictions_tem += [y_concat_prob]
            predictions_array = torch.cat(predictions_tem).detach().cpu().numpy()
            if return_hier_pred:
                hier_pred = torch.cat(hier_pred_tem).detach().cpu().numpy()
            self.model.train()
            if not return_probs_only:
                if return_hier_pred:
                    return torch.cat(ground_truth).detach().cpu().numpy(), predictions_array, hier_pred
                else:
                    return torch.cat(ground_truth).detach().cpu().numpy(), predictions_array
            else:
                if return_hier_pred:
                    return predictions_array, hier_pred
                else:
                    return predictions_array

    @torch.no_grad()
    def make_predictions(self, tag, return_pred_only=False,
                         thre=(0.08, 0.08, 0.08, 0.08), upper_bound=(3, 4, 17, 18), lower_bound=3,
                         boundary=([0, 100], [100, 781], [786, 2706], [2706, 3474])):
        self.model.eval()
        predictions_tem = []
        if tag == 'test':
            _dataloader = self.dataloader_test
            for data_tensors in _dataloader:
                x = data_tensors
                if self.use_cuda and torch.cuda.is_available():
                    x = x.cuda()
                y_concat_pred = regularized_pred(self.model.get_concat_probs(x).detach().cpu().numpy(),
                                                 thre=thre,
                                                 upper_bound=upper_bound, lower_bound=lower_bound, boundary=boundary)
                predictions_tem += [y_concat_pred]
            predictions_array = np.concatenate(predictions_tem)
            return predictions_array
        else:
            ground_truth = []
            if tag == 'train':
                _dataloader = self.dataloader_train
            elif tag == 'val':
                _dataloader = self.dataloader_val
            elif tag == 'all':
                _dataloader = self.dataloader_all
            else:
                raise ValueError('Invalid tag!')
            for data_tensors in _dataloader:
                if not return_pred_only:
                    x, y0, y1, y2, y3, y4 = data_tensors
                    if self.use_cuda and torch.cuda.is_available():
                        x = x.cuda()
                        y0 = y0.cuda()
                        y1 = y1.cuda()
                        y2 = y2.cuda()
                        y3 = y3.cuda()
                        y4 = y4.cuda()
                    ground_truth += [torch.cat((y0.long(),
                                                y1.long(),
                                                F.one_hot(y2, num_classes=6).squeeze()[:, 1:].long(),
                                                y3.long(),
                                                y4.long()), dim=1)]
                else:
                    x, _, _, _, _, _ = data_tensors
                    if self.use_cuda and torch.cuda.is_available():
                        x = x.cuda()
                y_concat_pred = regularized_pred(self.model.get_concat_probs(x).detach().cpu().numpy(),
                                                 thre=thre,
                                                 upper_bound=upper_bound, lower_bound=lower_bound, boundary=boundary)
                predictions_tem += [y_concat_pred]
            predictions_array = np.concatenate(predictions_tem)
            self.model.train()
            if not return_pred_only:
                return torch.cat(ground_truth).detach().cpu().numpy(), predictions_array
            else:
                return predictions_array

    @torch.no_grad()
    def compute_hier_acc(self, tag):
        ground_truth = []
        hier_pred_tem = []
        if tag == 'train':
            _dataloader = self.dataloader_train
        elif tag == 'val':
            _dataloader = self.dataloader_val
        elif tag == 'all':
            _dataloader = self.dataloader_all
        else:
            raise ValueError('Invalid tag!')
        for data_tensors in _dataloader:
            x, _, _, y2, _, _ = data_tensors
            if self.use_cuda and torch.cuda.is_available():
                x = x.cuda()
                y2 = y2.cuda()
            gt_oh = F.one_hot(y2, num_classes=6).squeeze()[:, 1:].long()
            ground_truth += [torch.sum(gt_oh, dim=-1)]
            if self.model.hierarchical:
                _, y_group_pred = self.model.get_concat_probs(x, return_hier_pred=True)
                hier_pred_tem += [y_group_pred]
            else:
                y_concat_prob = self.model.get_concat_probs(x, return_hier_pred=False)
                hier_pred_tem += [torch.sum(y_concat_prob[:, 781:786], dim=-1)]
        if self.model.hierarchical:
            return np.mean(torch.cat(ground_truth).detach().cpu().numpy() ==
                           np.where(torch.cat(hier_pred_tem).detach().cpu().numpy() == 1)[1])
        else:
            return np.mean(torch.cat(ground_truth).detach().cpu().numpy() ==
                           torch.cat(hier_pred_tem).detach().cpu().numpy())

    @torch.no_grad()
    def compute_accuracy(self, tag,
                         thre=(0.08, 0.08, 0.08, 0.08), upper_bound=(3, 4, 17, 18), lower_bound=3,
                         boundary=([0, 100], [100, 781], [786, 2706], [2706, 3474])):
        y_true, y_pred = self.make_predictions(tag=tag, thre=thre,
                                               upper_bound=upper_bound, lower_bound=lower_bound, boundary=boundary)
        f_beta = [fbeta_score(y_true[i, :], y_pred[i, :], beta=2) for i in range(y_true.shape[0])]
        return sum(f_beta) / len(f_beta)

    @torch.no_grad()
    def accuracy_history_plot(self, epochs_override=None):
        len_ticks = len(self.accuracy_history_train)
        if epochs_override is None:
            x_axis = np.linspace(1, self.epochs, len_ticks)
        else:
            x_axis = np.linspace(1, epochs_override, len_ticks)
        plt.figure()
        if self.train_val:
            assert (len(self.accuracy_history_train) == len(self.accuracy_history_val))
            plt.plot(x_axis, self.accuracy_history_train, label='Training set')
            plt.plot(x_axis, self.accuracy_history_val, label='Validation set')
            plt.legend()
        else:
            plt.plot(x_axis, self.accuracy_history_train)

        plt.xlabel('Number of epochs')
        plt.ylabel('Accuracy')
        plt.show()

    @torch.no_grad()
    def get_features(self, tag, return_gt=True):
        self.model.eval()
        features_tem = []

        if tag == 'test':
            _dataloader = self.dataloader_test
            for data_tensors in _dataloader:
                x = data_tensors
                if self.use_cuda and torch.cuda.is_available():
                    x = x.cuda()
                y_concat_prob = self.model.inference(x)
                features_tem += [y_concat_prob]
            features_array = torch.cat(features_tem).detach().cpu().numpy()
            self.model.train()
            return features_array
        else:
            ground_truth = []
            if tag == 'train':
                _dataloader = self.dataloader_train
            elif tag == 'val':
                _dataloader = self.dataloader_val
            elif tag == 'all':
                _dataloader = self.dataloader_all
            else:
                raise ValueError('Invalid tag!')
            for data_tensors in _dataloader:
                if return_gt:
                    x, y0, y1, y2, y3, y4 = data_tensors
                    if self.use_cuda and torch.cuda.is_available():
                        x = x.cuda()
                        y0 = y0.cuda()
                        y1 = y1.cuda()
                        y2 = y2.cuda()
                        y3 = y3.cuda()
                        y4 = y4.cuda()
                    ground_truth += [torch.cat((y0.long(),
                                                y1.long(),
                                                F.one_hot(y2, num_classes=6).squeeze()[:, 1:].long(),
                                                y3.long(),
                                                y4.long()), dim=1)]
                else:
                    x, _, _, _, _, _ = data_tensors
                    if self.use_cuda and torch.cuda.is_available():
                        x = x.cuda()
                y_concat_prob = self.model.inference(x)
                features_tem += [y_concat_prob]
            features_array = torch.cat(features_tem).detach().cpu().numpy()
            self.model.train()
            return torch.cat(ground_truth).detach().cpu().numpy(), features_array
