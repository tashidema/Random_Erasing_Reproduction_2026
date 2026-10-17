"""Independent RGB random filling on a ToTensor image BEFORE normalization.
Channel independence and transform order are explicit implementation choices.
"""
import random
import math
import torch

class RandomErasing:
    def __init__(self, probability=0.5, sl=0.02, sh=0.20, r1=0.3):
        self.probability, self.sl, self.sh, self.r1 = probability, sl, sh, r1

    def __call__(self, img):
        if random.random() >= self.probability:
            return img
        height, width = img.shape[-2:]
        for _ in range(100):
            area = random.uniform(self.sl, self.sh) * height * width
            ratio = random.uniform(self.r1, 1 / self.r1)
            h = round(math.sqrt(area * ratio))
            w = round(math.sqrt(area / ratio))
            if 0 < h < height and 0 < w < width:
                top = random.randint(0, height - h)
                left = random.randint(0, width - w)
                patch = img[:, top:top+h, left:left+w]
                patch.copy_(torch.rand_like(patch))
                break
        return img
