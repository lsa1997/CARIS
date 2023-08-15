import numpy as np

import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class Resize(object):
    def __init__(self, h, w, eval_mode=False):
        self.h = h
        self.w = w
        self.eval_mode = eval_mode

    def __call__(self, image, target):
        image = F.resize(image, (self.h, self.w))
        # If size is a sequence like (h, w), the output size will be matched to this.
        # If size is an int, the smaller edge of the image will be matched to this number maintaining the aspect ratio
        if not self.eval_mode:
            if isinstance(target, list):
                target_new = []
                for _target in target:
                    target_new.append(F.resize(_target, (self.h, self.w), interpolation=F.InterpolationMode.NEAREST))
                target = target_new
            else:
                target = F.resize(target, (self.h, self.w), interpolation=F.InterpolationMode.NEAREST)
        return image, target

class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        if isinstance(target, list):
            target_new = []
            for _target in target:
                target_new.append(torch.as_tensor(np.asarray(_target).copy(), dtype=torch.int64))
            target = target_new
        else:
            target = torch.as_tensor(np.asarray(target).copy(), dtype=torch.int64)
        return image, target

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target

