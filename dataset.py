import torch
import numpy as np
from torch.utils.data import Dataset
from preprocess import get_patient_stft_data
from utils import N_FREQ_BINS

class SienaSTFTDataset(Dataset):
    def __init__(self, target_patients):
        X_list     = []
        y_list     = []
        group_list = []

        for patient in target_patients:
            X_p, y_p = get_patient_stft_data(patient)
            if X_p is None:
                continue

            # Standardize per patient
            p_mean = X_p.mean(axis=(0, 2, 3), keepdims=True)
            p_std  = X_p.std(axis=(0, 2, 3),  keepdims=True) + 1e-6
            X_p    = (X_p - p_mean) / p_std

            X_list.append(X_p)
            y_list.append(y_p)
            group_list.append(np.full(len(y_p), patient))

        if X_list:
            self.X      = np.concatenate(X_list, axis=0)
            self.y      = np.concatenate(y_list, axis=0)
            self.groups = np.concatenate(group_list, axis=0)
        else:
            self.X      = np.empty((0, 18, 1, N_FREQ_BINS), dtype=np.float32)
            self.y      = np.empty((0,), dtype=np.float32)
            self.groups = np.empty((0,), dtype=object)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])