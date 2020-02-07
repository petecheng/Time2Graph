# -*- coding: utf-8 -*-
import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from ...utils.base_utils import Debugger
"""
    utils for deep models.
"""


def latent_loss(z_mean, z_std):
    mean_2 = z_mean * z_mean
    std_2 = z_std * z_std
    return 0.5 * torch.mean(mean_2 + std_2 - torch.log(std_2) - 1)


class DeepDataloader(DataLoader):
    def __init__(self, *args, **kwargs):
        super(DeepDataloader, self).__init__(*args, **kwargs)


class DeepDataset(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, item):
        return self.x[item], self.y[item]

    def __len__(self):
        return len(self.y)


def train_RNNs(epoch, dataloader, rnn, criterion, optimizer, debug, gpu_enable):
    rnn.train()
    for i, (sequences, target) in enumerate(dataloader, 0):
        sequences = sequences.double()
        if gpu_enable:
            sequences = sequences.cuda()
            target = target.cuda()
        sequences = Variable(sequences)
        target = Variable(target)
        output = rnn(sequences)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % int(len(dataloader) / 10 + 1) == 0:
            Debugger.debug_print('[{}][{}][{}], Loss: {}'.format(
                epoch, i, len(dataloader), loss.item()), debug=debug)


def train_VAE(epoch, dataloader, vae, criterion, optimizer, debug, gpu_enable):
    vae.train()
    for i, (sequences, target) in enumerate(dataloader, 0):
        optimizer.zero_grad()
        sequences = sequences.double()
        if gpu_enable:
            sequences = sequences.cuda()
            target = target.cuda()
        sequences = Variable(sequences)
        output = vae(sequences)
        loss = criterion(output, target) + latent_loss(vae.z_mean, vae.z_sigma)
        loss.backward()
        optimizer.step()
        if i % int(len(dataloader) / 10 + 1) == 0:
            Debugger.debug_print('[{}][{}][{}], Loss: {}'.format(
                epoch, i, len(dataloader), loss.item()), debug=debug)


def test_DeepModels(dataloader, rnn, criterion, debug, gpu_enable):
    for th in range(5, 20, 1):
        test_loss = 0
        correct = 0
        rnn.eval()
        y_pred, y_test = [], []
        th = th / 20
        for i, (sequences, target) in enumerate(dataloader, 0):
            rnn.zero_grad()
            sequences = sequences.double()
            if gpu_enable:
                sequences = sequences.cuda()
                target = target.cuda()
            sequences = Variable(sequences)
            target = Variable(target)
            output = rnn(sequences)
            test_loss += criterion(output, target).item()
            # pred = F.softmax(output, dim=1).data.max(1, keepdim=True)[1]
            pred = F.softmax(output, dim=1)[:, 1].data.cpu().numpy()
            print(pred)
            tmp = np.zeros(len(pred))
            tmp[pred >= th] = 1
            # y_pred += list(pred.cpu().numpy())
            y_pred += list(tmp)
            y_test += list(target.cpu().numpy())
            # correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        test_loss /= len(dataloader.dataset)
        y_pred, y_test = np.array(y_pred, dtype=np.int).reshape(-1), np.array(y_test, dtype=np.int).reshape(-1)
        accu = accuracy_score(y_true=y_test, y_pred=y_pred)
        prec = precision_score(y_true=y_test, y_pred=y_pred)
        recall = recall_score(y_true=y_test, y_pred=y_pred)
        f1 = f1_score(y_true=y_test, y_pred=y_pred)
        Debugger.info_print('res: accu {:.4f}, prec {:.4f}, recall {:.4f}, f1 {:.4f}'.format(
            accu, prec, recall, f1
        ))
        Debugger.debug_print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            test_loss, correct, len(dataloader.dataset),
            100. * correct / len(dataloader.dataset)), debug=debug)
