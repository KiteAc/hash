import pickle as pkl
import numpy as np
import torch

pkl_dir2 = "/home/sun/project/dataset/ESC-50-master/pth/validation128mel4.pkl"
data = []
targets = []
values = []
with open(pkl_dir2, "rb") as f:
    data = pkl.load(f)
    for i in range(0, len(data)):
        targets.append(data[i]['target'])
        data.append(data[i]['values'])

train_labels = np.array(targets, dtype=np.float32)
train_data = data
labels_set = set(train_labels)
print(train_labels)
label_to_indices = {label: np.where(train_labels == label)[0] for label in labels_set}

print(label_to_indices)

print("{}training128mel{}.pkl".format(pkl_dir2, 1))

tuplexx = [1, 2]
for i in range(len(tuplexx)):
    tuplexx[i] = torch.FloatTensor(tuplexx[i])
print(tuplexx)









