import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
import numpy as np
from torch.optim import lr_scheduler
import torch.optim as optim

from networks import EmbeddingNet, TripletNet
from losses import TripletLoss
import datasetaug
from torchvision import models
from trainer import fit
from utils import freeze_model, list_trainable, del_last_layers, save, load, create_embeddings
from datasets import TripletCifar
from datasets import TripletMNIST
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

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def freeze_model(model):
    for params in model.parameters():
        params.requires_grad = False



if __name__ == "__main__":
    margin = 2.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    pkl_dir = "/home/sun/project/dataset/ESC-50-master/pth/"
    train_transforms = datasetaug.MelSpectrogram(128, "train", "ESC")

    embedding_net = models.resnet50(pretrained =True)
    embedding_net.classifier = Identity()
    model = TripletNet(embedding_net)
    model.cuda()
    loss_fn = TripletLoss(margin)
    lr = 1e-3
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
    n_epochs = 10
    log_interval = 500


    for i in range(5):
        train_dataset = datasetaug.AudioDataset( "{}training128mel{}.pkl".format(pkl_dir, i+1), "ESC", transforms = train_transforms)
        test_dataset = datasetaug.AudioDataset( "{}validation128mel{}.pkl".format(pkl_dir, i+1), "ESC")
        triplet_train_dataset = TripletCifar(train_dataset, "train")
        triplet_test_dataset = TripletCifar(test_dataset, "test")

        triplet_train_loader = torch.utils.data.DataLoader(triplet_train_dataset, batch_size=64, shuffle=True)
        triplet_test_loader = torch.utils.data.DataLoader(triplet_test_dataset, batch_size=64, shuffle=False)


        fit(triplet_train_loader, triplet_test_loader, model, loss_fn, optimizer, scheduler, n_epochs, device, log_interval)
    torch.save(model, './triplet_resnet50_2.mdl')





