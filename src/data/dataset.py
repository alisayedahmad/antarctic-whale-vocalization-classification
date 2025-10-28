import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple


class WhaleVocalizationDataset(Dataset):
    """PyTorch Dataset for whale vocalizations."""
    
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        transform=None
    ):
        self.X = X
        self.y = y
        self.transform = transform
        
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.X[idx]
        label = self.y[idx]
        
        if self.transform:
            sample = self.transform(sample)
        
        if len(sample.shape) == 2:
            sample = sample.flatten()
        
        sample = torch.FloatTensor(sample)
        label = torch.LongTensor([label])[0]
        
        return sample, label


class BalancedBatchSampler:
    """Sampler for balanced batches."""
    
    def __init__(
        self,
        y: np.ndarray,
        batch_size: int,
        num_batches: int = None
    ):
        self.y = y
        self.batch_size = batch_size
        self.num_batches = num_batches or len(y) // batch_size
        
        self.classes = np.unique(y)
        self.class_indices = {}
        
        for cls in self.classes:
            self.class_indices[cls] = np.where(y == cls)[0]
    
    def __iter__(self):
        for _ in range(self.num_batches):
            batch = []
            samples_per_class = self.batch_size // len(self.classes)
            
            for cls in self.classes:
                indices = np.random.choice(
                    self.class_indices[cls],
                    size=samples_per_class,
                    replace=True
                )
                batch.extend(indices)
            
            np.random.shuffle(batch)
            yield batch
    
    def __len__(self) -> int:
        return self.num_batches
