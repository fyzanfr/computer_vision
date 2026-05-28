import torch
import numpy as np 

def cutout(length=8):
    def _fn(img):
        h, w = img.size(0), img.size(1)
        mask = np.ones((h, w), np.float32)

        x = np.random.randint(h) 
        y = np.random.randint(w)

        y1 = np.clip(y - length // 2, 0, h) # bottom-half
        y2 = np.clip(y + length // 2, 0, h) # top-half
        x1 = np.clip(x - length // 2, 0, w) # left-half
        x2 = np.clip(x + length // 2, 0, w) # right-half

        mask[y1:y2, x1:x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask
        return img
