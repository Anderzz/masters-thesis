import numpy as np
import albumentations as A
from albumentations.core.transforms_interface import DualTransform, ImageOnlyTransform


class Blackout(DualTransform):
    def __init__(self, probability=0.5, always_apply=False, p=0.5):
        super(Blackout, self).__init__(always_apply, p)
        self.probability = probability

    def apply(self, image, **params):
        # Randomize parameters for each call
        x = np.random.randint(50, 200)
        y = np.random.randint(100, 200)
        width = np.random.randint(25, 65)
        height = np.random.randint(25, 65)

        if np.random.rand() > self.probability:
            return image  # Return unchanged image if not within probability

        # Apply blackout
        image[y : y + height, x : x + width] = 0
        return image

    def get_transform_init_args_names(self):
        return ("probability",)
