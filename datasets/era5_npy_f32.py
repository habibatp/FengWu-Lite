import os
import numpy as np
import torch
from torch.utils.data import Dataset


class ERA5Dataset(Dataset):
    def __init__(
        self,
        data_dir,
        split="train",
        input_steps=2,
        target_steps=4,   # 4 steps = 1 jour avec pas de 6h
        sample_stride=1,
        **kwargs
    ):
        super().__init__()

        self.data_dir = data_dir
        self.split = split
        self.input_steps = input_steps
        self.target_steps = target_steps
        self.sample_stride = sample_stride

        self.single_vars = ['u10', 'v10', 't2m', 'msl']
        self.multi_vars = ['z', 'q', 'u', 'v', 't']

        self.levels = [
            1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 125, 150, 175,
            200, 225, 250, 300, 350, 400, 450, 500, 550, 600, 650,
            700, 750, 775, 800, 825, 850, 875, 900, 925, 950, 975, 1000
        ]

        self.years = kwargs.get("years", None)

        all_files = os.listdir(data_dir)
        valid_files = []
        for f in all_files:
            if not f.endswith(".npy"):
                continue
            if self.years is not None:
                try:
                    year = int(f.split("_")[0])
                    if year not in self.years:
                        continue
                except ValueError:
                    pass
            valid_files.append(f)

        self.files = sorted(valid_files)

        self.total_steps = self.input_steps + self.target_steps

        if len(self.files) < self.total_steps:
            raise ValueError(
                f"Pas assez de fichiers .npy. "
                f"Trouvé {len(self.files)}, besoin au minimum de {self.total_steps}."
            )

    def __len__(self):
        return max(0, len(self.files) - self.total_steps * self.sample_stride + 1)

    def get_maxidx(self):   # 🔥 AJOUT IMPORTANT
        return len(self)
    def get_target(self, idx):
        idx = int(idx)
        targets = []

        for i in range(self.target_steps):
            file_idx = idx + self.input_steps + i * self.sample_stride
            file_idx = min(file_idx, len(self.files) - 1)
            targets.append(self.load_file(self.files[file_idx]))

        targets = np.stack(targets, axis=0)  # [target_steps, 189, 64, 64]
        
        # Normalisation locale pour la cohérence avec __getitem__
        mean = targets.mean(axis=(0, 2, 3), keepdims=True)
        std = targets.std(axis=(0, 2, 3), keepdims=True)
        std[std == 0] = 1.0
        return (targets - mean) / std
    

    def get_meanstd(self):   # 🔥 AJOUT IMPORTANT
        mean = np.zeros((189,), dtype=np.float32)
        std = np.ones((189,), dtype=np.float32)
        return mean, std
    def load_file(self, filename):
        data = np.load(os.path.join(self.data_dir, filename)).astype(np.float32)

        if data.shape != (189, 64, 64):
            raise ValueError(f"Shape invalide pour {filename}: {data.shape}, attendu (189,64,64)")

        return data

    def __getitem__(self, idx):
        seq = []

        for i in range(self.total_steps):
            file_idx = idx + i * self.sample_stride
            data = self.load_file(self.files[file_idx])
            seq.append(data)

        seq = np.stack(seq, axis=0)  # [T, C, H, W]

        # 🔥 AJOUT IMPORTANT: Normalisation par sample pour éviter la perte NaN
        # Idéalement, il faut utiliser la moyenne/std globale du dataset, mais en attendant:
        mean = seq.mean(axis=(0, 2, 3), keepdims=True)
        std = seq.std(axis=(0, 2, 3), keepdims=True)
        std[std == 0] = 1.0
        seq = (seq - mean) / std
        x = seq[:self.input_steps]       # [2, 189, 64, 64]
        y = seq[self.input_steps:]       # [target_steps, 189, 64, 64]

        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()

        return x, y
       

# 🔥 IMPORTANT
era5_npy_f32 = ERA5Dataset