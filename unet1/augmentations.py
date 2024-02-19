import albumentations as A
import numpy as np


class Rotate:
    def __init__(self, limit=45, p=0.5):
        self.transform = A.Rotate(limit=limit, p=p)

    def apply(self, image):
        return self.transform(image=image)["image"]


class GammaTransform:
    def __init__(self, gamma=1.5, p=0.5):
        self.transform = A.RandomGamma(gamma_limit=(gamma, gamma), p=p)

    def apply(self, image):
        return self.transform(image=image)["image"]


class Blackout:
    def __init__(self, num_rectangles=1, p=0.5):
        self.transform = A.Cutout(
            num_holes=num_rectangles, max_h_size=50, max_w_size=50, fill_value=0, p=p
        )

    def apply(self, image):
        return self.transform(image=image)["image"]
