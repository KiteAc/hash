from torch.utils.data import *

import torchvision
import pandas as pd
import numpy as np
import pickle
import torch
import librosa
import torchaudio
import random
from PIL import Image


class MelSpectrogram(object):
    def __init__(self, bins, mode, dataset):
        self.window_length = [25, 50, 100]
        self.hop_length = [10, 25, 50]
        self.fft = 4410 if dataset == "ESC" else 2205
        self.melbins = bins
        self.mode = mode
        self.sr = 44100 if dataset == "ESC" else 22050
        self.length = 1500 if dataset == "GTZAN" else 250

    def __call__(self, value):
        sample = value
        limits = ((-2, 2), (0.9, 1.2))

        if self.mode == "train":
            pitch_shift = np.random.randint(limits[0][0], limits[0][1] + 1)
            time_stretch = np.random.random() * (limits[1][1] - limits[1][0]) + limits[1][0]
            new_audio = librosa.effects.time_stretch(librosa.effects.pitch_shift(sample, self.sr, pitch_shift),
                                                     time_stretch)
        else:
            pitch_shift = 0
            time_stretch = 1
            new_audio = sample
        specs = []
        for i in range(len(self.window_length)):
            clip = torch.Tensor(new_audio)

            window_length = int(round(self.window_length[i] * self.sr / 1000))
            hop_length = int(round(self.hop_length[i] * self.sr / 1000))
            spec = torchaudio.transforms.MelSpectrogram(sample_rate=self.sr, n_fft=self.fft, win_length=window_length,
                                                        hop_length=hop_length, n_mels=self.melbins)(clip)
            eps = 1e-6
            spec = spec.numpy()
            spec = np.log(spec + eps)
            spec = np.asarray(torchvision.transforms.Resize((128, self.length))(Image.fromarray(spec)))
            specs.append(spec)
        specs = np.array(specs).reshape(-1, 128, self.length)
        specs = torch.Tensor(specs)
        return specs


class AudioDataset(Dataset):
    def __init__(self, pkl_dir, dataset_name, mode = "test", transforms=None,):
        self.mode = mode
        self.transforms = transforms
        self.data = []
        self.targets = []
        self.length = 1500 if dataset_name == "GTZAN" else 250
        with open(pkl_dir, "rb") as f:
            entry = pickle.load(f)
            for i in range(0, len(entry)):
                self.targets.append(entry[i]['target'])
                self.data.append(entry[i]['values'])

            self.data = np.vstack(self.data).reshape(-1, 3, 128, 250)
            self.data = self.data.transpose((0, 2, 3, 1))

    def __len__(self):

        return len(self.data)

    def __getitem__(self, idx):
        if idx >= len(self.data):

            new_idx = idx - len(self.data)
            entry = self.data[new_idx]

            if self.transforms:
                values = self.transforms(entry["audio"])

        else:
            entry = self.data[idx]
            values = torch.Tensor(entry.reshape(-1, 128, self.length))
        targets = torch.LongTensor(self.targets)

        return (values, targets)


def fetch_dataloader(pkl_dir, dataset_name, batch_size, num_workers, mode):
    # MelSpectrogram的实例
    transforms = MelSpectrogram(128, mode, dataset_name)
    # AudioDataset的实例
    dataset = AudioDataset(pkl_dir, dataset_name, transforms=transforms)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers)
    return dataloader




