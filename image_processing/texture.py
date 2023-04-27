import numpy as np
from PIL import Image
import skimage as sk


class GaussianNoise():
    parameters = {"scale":True}

    def __init__(self, scale, **kwargs):
        self.scale = scale

    def __call__(self, x):
        x = np.array(x) / 255.
        return np.clip(x + np.random.normal(size=x.shape, scale=self.scale), 0, 1) * 255

    def gaussian_noise(x, severity=1):
        c = [.08, .12, 0.18, 0.26, 0.38][severity - 1]

        x = np.array(x) / 255.
        return np.clip(x + np.random.normal(size=x.shape, scale=c), 0, 1) * 255


class ShotNoise():
    parameters = {"lam_weight":True}

    def __init__(self, lam_weight, **kwargs):
        self.lam_weigth = lam_weight #must be Int between 0 and 100

    def __call__(self, x):
        x = np.array(x) / 255.
        return np.clip(np.random.poisson(x * self.lam_weigth) / float(self.lam_weigth), 0, 1) * 255

    def shot_noise(x, severity=1):
        c = [60, 25, 12, 5, 3][severity - 1]

        x = np.array(x) / 255.
        return np.clip(np.random.poisson(x * c) / float(c), 0, 1) * 255


class ImpulseNoise():
    parameters = {"amount":True}

    def __init__(self, amount, **kwargs):
        self.amount = amount

    def __call__(self, x):
        x = sk.util.random_noise(np.array(x) / 255., mode='s&p', amount=self.amount)
        return np.clip(x, 0, 1) * 255

    def impulse_noise(x, severity=1):
        c = [.03, .06, .09, 0.17, 0.27][severity - 1]

        x = sk.util.random_noise(np.array(x) / 255., mode='s&p', amount=c)
        return np.clip(x, 0, 1) * 255


class SpekleNoise():
    parameters = {"scale":True}

    def __init__(self, scale, **kwargs):
        self.scale = scale

    def __call__(self, x):
        x = np.array(x) / 255.
        return np.clip(x + x * np.random.normal(size=x.shape, scale=self.scale), 0, 1) * 255

    def speckle_noise(x, severity=1):
        c = [.15, .2, 0.35, 0.45, 0.6][severity - 1]

        x = np.array(x) / 255.
        return np.clip(x + x * np.random.normal(size=x.shape, scale=c), 0, 1) * 255