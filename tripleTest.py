from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from torchvision import models


import pickle
import numpy as np
import torch.nn as nn
import torch
import datasetaug

class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2, x3):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        return output1, output2, output3

    def get_embedding(self, x):
        return self.embedding_net(x)

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

def load(model_name):
  with open(model_name+'.embs', 'rb') as file:
    features, targets = pickle.load(file)
    return (features, targets)

def search_knn_accuracies(k_range, features, targets):
  acc = []
  for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(features['train'], targets['train'])

    predict = knn.predict(features['test'][:300,:])
    score = metrics.accuracy_score(targets['test'][:300], predict)
    print('K value: %d, accuracy: %0.7f' %(k, score))
    acc.append(score)
  return acc


def create_embeddings(model, train_data_loader, test_dataloader , embedding_size):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    features = {}
    targets = {}
    model.to(device)
    features['train'] = np.empty([0, embedding_size])
    targets['train'] = np.empty([0, ])

    features['test'] = np.empty([0, embedding_size])
    targets['test'] = np.empty([0, ])

    for i, (images, target) in enumerate(train_data_loader):
        images = images.to(device)
        target = target.to(device)

        try:
            output = model(images).cpu().numpy()
            features['train'] = np.append(features['train'], output, axis=0)
            targets['train'] = np.append(targets['train'], target.cpu(), axis=0)
        except:
            print('error occured: ')
            return (None, None)

        if i % 100 == 0:
            print(i)

    for i, (images, target) in enumerate(test_data_loader):
        images = images.to(device)
        target = target.to(device)

        output = model(images).cpu().numpy()
        features['test'] = np.append(features['test'], output, axis=0)
        targets['test'] = np.append(targets['test'], target.cpu(), axis=0)

        if i % 100 == 0:
            print(i)
    return (features, targets)

if __name__ == "__main__":

    k_range = range(25, 30)

    resnet18 = models.resnet50(pretrained=True)
    resnet18.fc = Identity()

    model = TripletNet(resnet18)
    model_emb = model.embedding_net
    pkl_dir = "/home/sun/project/dataset/ESC-50-master/pth/"
    train_dataset = datasetaug.AudioDataset("{}training128mel{}.pkl".format(pkl_dir,1), "ESC",transforms="None")
    test_dataset = datasetaug.AudioDataset("{}validation128mel{}.pkl".format(pkl_dir, 1), "ESC", transforms="None")

    train_data_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=128, num_workers=8)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, shuffle = False, batch_size = 128, num_workers = 8)
    feature, target = create_embeddings(model_emb, train_data_loader,test_data_loader, 512)

    #acc = search_knn_accuracies(k_range, feature, target)
