

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np
import datasetaug
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import datasets

import torch
from torch.optim import lr_scheduler
import torch.optim as optim
from torch.autograd import Variable


import numpy as np
cuda = torch.cuda.is_available()

from torch.utils.data.sampler import BatchSampler

import matplotlib
import matplotlib.pyplot as plt


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class BinaryLayer(nn.Module):
    def __init__(self):
        super(BinaryLayer, self).__init__()
        self.binaryLayer = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 32)
        )

    def forward(self, x):
        return self.binaryLayer(x)

from torch.utils.data.sampler import BatchSampler

class BalancedBatchSampler1(BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, labels, n_classes, n_samples):
        self.labels = np.array(labels, dtype=np.float32)
        # label
        self.labels_set = list(set(self.labels))
        # label jian li suo yin
        self.label_to_indices = {label: np.where(self.labels == label)[0]
                                 for label in self.labels_set}
        # da po suo yin de shun xu
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])

        # dict(0.0:0, 1.0:0,...,49:0)
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.n_dataset = len(self.labels)
        # 25 * 50 = 1250
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < self.n_dataset:
            # sui ji fan hui n_classes ge label
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return self.n_dataset // self.batch_size

def fit(train_loader, val_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval, metrics=[],
        start_epoch=0):
    """
    Loaders, model, loss function and metrics should work together for a given task,
    i.e. The model should be able to process data output of loaders,
    loss function should process target output of loaders and outputs from the model

    Examples: Classification: batch loader, classification model, NLL loss, accuracy metric
    Siamese network: Siamese loader, siamese model, contrastive loss
    Online triplet learning: batch loader, embedding model, online triplet loss
    """
    for epoch in range(0, start_epoch):
        scheduler.step()

    for epoch in range(start_epoch, n_epochs):
        scheduler.step()

        # Train stage
        train_loss, metrics = train_epoch(train_loader, model, loss_fn, optimizer, cuda, log_interval, metrics)

        message = 'Epoch: {}/{}. Train set: Average loss: {:.4f}'.format(epoch + 1, n_epochs, train_loss)
        for metric in metrics:
            message += '\t{}: {}'.format(metric.name(), metric.value())

        val_loss, metrics = test_epoch(val_loader, model, loss_fn, cuda, metrics)
        val_loss /= len(val_loader)

        message += '\nEpoch: {}/{}. Validation set: Average loss: {:.4f}'.format(epoch + 1, n_epochs,
                                                                                 val_loss)
        for metric in metrics:
            message += '\t{}: {}'.format(metric.name(), metric.value())

        print(message)



def freeze_model(model):
  for params in model.parameters():
    params.requires_grad=False

def train_epoch(train_loader, model, loss_fn, optimizer, cuda, log_interval, metrics):
    for metric in metrics:
        metric.reset()

    model.train()
    losses = []
    total_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):

        target = target if len(target) > 0 else None

        if not type(data) in (tuple, list):
            data = (data,)
        if cuda:
            data = tuple(d.cuda() for d in data)


            if target is not None:
                target = target.cuda()


        optimizer.zero_grad()
        outputs = model(*data)


        if type(outputs) not in (tuple, list):
            outputs = (outputs,)

        loss_inputs = outputs
        if target is not None:
            target = (target,)
            loss_inputs += target

        loss_outputs = loss_fn(*loss_inputs)
        loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
        losses.append(loss.item())
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        for metric in metrics:
            metric(outputs, target, loss_outputs)

        if batch_idx % log_interval == 0:
            message = 'Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                batch_idx * len(data[0]), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), np.mean(losses))
            for metric in metrics:
                message += '\t{}: {}'.format(metric.name(), metric.value())

            print(message)
            losses = []

    total_loss /= (batch_idx + 1)
    return total_loss, metrics


def test_epoch(val_loader, model, loss_fn, cuda, metrics):
    with torch.no_grad():
        for metric in metrics:
            metric.reset()
        model.eval()
        val_loss = 0
        for batch_idx, (data, target) in enumerate(val_loader):
            target = target if len(target) > 0 else None
            if not type(data) in (tuple, list):
                data = (data,)
            if cuda:
                data = tuple(d.cuda() for d in data)
                if target is not None:
                    target = target.cuda()

            outputs = model(*data)

            if type(outputs) not in (tuple, list):
                outputs = (outputs,)
            loss_inputs = outputs
            if target is not None:
                target = (target,)
                loss_inputs += target

            loss_outputs = loss_fn(*loss_inputs)
            loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
            val_loss += loss.item()

            for metric in metrics:
                metric(outputs, target, loss_outputs)

    return val_loss, metrics


# dataset path
pkl_dir = "/home/sun/project/dataset/ESC-50-master/pth/"
train_transforms = datasetaug.MelSpectrogram(128, "train", "ESC")

torch.cuda.set_device(1)
resnet50 = models.resnet50(pretrained=True)
resnet50.fc = BinaryLayer()
resnet50.cuda()

for i in range(5):
    train_dataset = datasetaug.AudioDataset("{}training128mel{}.pkl".format(pkl_dir, i + 1), "ESC",
                                            transforms=train_transforms)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8)
    test_dataset = datasetaug.AudioDataset("{}validation128mel{}.pkl".format(pkl_dir, i + 1), "ESC", mode= "train")
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=8)

    train_batch_sampler = BalancedBatchSampler1(train_dataset.targets, n_classes=50, n_samples=5)
    test_batch_sampler = BalancedBatchSampler1(test_dataset.targets, n_classes=50, n_samples=5)

    online_train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=train_batch_sampler)
    online_test_loader = torch.utils.data.DataLoader(test_dataset, batch_sampler=test_batch_sampler)



    for i ,(images, target) in enumerate(online_train_loader):
        out = resnet50(images.cuda())
        print(torch.sign(out))







