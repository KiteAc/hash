import numpy as np
import datasetaug

n_class = 50
n_samples = 25
pkl_dir = "/home/sun/project/dataset/ESC-50-master/pth/"

train_dataset = datasetaug.AudioDataset("{}training128mel{}.pkl".format(pkl_dir, 1), "ESC",transforms=None)


label = np.zeros(100, dtype= float)
for l in set(label):
    print(l)

