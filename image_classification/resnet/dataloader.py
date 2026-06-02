import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR100
import multiprocessing

CIFAR100_MEAN = [0.5071, 0.4867, 0.4408]
CIFAR100_STD = [0.2675, 0.2565, 0.2761]
NUM_CLASSES = 100 



## Transformations

def cutout(length=8):
    def _fn(img):
        h, w = img.shape[:2]
        mask = np.ones((h, w), np.float32)

        x = np.random.randint(h) 
        y = np.random.randint(w)

        y1 = np.clip(y - length // 2, 0, h) # bottom-half
        y2 = np.clip(y + length // 2, 0, h) # top-half
        x1 = np.clip(x - length // 2, 0, w) # left-half
        x2 = np.clip(x + length // 2, 0, w) # right-half

        mask[y1:y2, x1:x2] = 0.
        mask = mask[:, :, np.newaxis]
        img = img * mask
        return img
    return _fn

def standardize(mean, std):
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    def _fn(img):
        return (img - mean) / std
    return _fn


def random_hflip(p=0.5):
    def _fn(img):
        if np.random.random() < p:
            return img[:, ::-1, :].copy()
        return img
    return _fn


def random_crop(pad=4):
    def _fn(img):
        h, w = img.shape[:2]
        padded = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
        y = np.random.randint(0, 2 * pad)
        x = np.random.randint(0, 2 * pad)
        return padded[y:y + h, x:x + w, :]
    return _fn


def compose(*fns):
    def _fn(img):
        for fn in fns:
            img = fn(img)
        return img
    return _fn



class CustomCIFAR100(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.ds = CIFAR100(root=root, train=train, download=True, transform=None)
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        img_pil, label = self.ds[i]
        img = np.array(img_pil, dtype=np.float32) / 255.0
        if self.transform is not None:
            img = self.transform(img)
        img = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float()
        return img, int(label)

use_cuda = torch.cuda.is_available()

def make_dataset(root="./data", batch_size=128, num_workers=2):
    train_tf = compose(random_hflip(), random_crop(pad=4), standardize(CIFAR100_MEAN, CIFAR100_STD))
    eval_tf = standardize(CIFAR100_MEAN, CIFAR100_STD)

    train_ds = CustomCIFAR100(root=root, train=True, transform=train_tf) 
    val_ds = CustomCIFAR100(root=root, train=False, transform=eval_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=use_cuda)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False, num_workers=num_workers, pin_memory=use_cuda)

    return train_loader, val_loader


if __name__ == "__main__":
    multiprocessing.set_start_method("fork", force=True)
    tl, vl = make_dataset()
    print(f"CIFAR-100 train batches: {len(tl)}, val batches: {len(vl)}")
    x, y = next(iter(tl))
    print(f"Batch shape: {x.shape},  labels: {y.shape},  classes: {y.unique().tolist()}")




