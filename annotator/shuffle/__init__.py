import random

import cv2
import numpy as np


def make_noise_disk(H, W, C, F):
    noise = np.random.uniform(low=0, high=1, size=((H // F) + 2, (W // F) + 2, C))
    noise = cv2.resize(noise, (W + 2 * F, H + 2 * F), interpolation=cv2.INTER_CUBIC)
    noise = noise[F: F + H, F: F + W]
    noise -= np.min(noise)
    noise /= np.max(noise)
    if C == 1:
        noise = noise[:, :, None]
    return noise


def img2mask(img, H, W, low=10, high=90):
    assert img.ndim == 3 or img.ndim == 2
    assert img.dtype == np.uint8

    if img.ndim == 3:
        y = img[:, :, random.randrange(0, img.shape[2])]
    else:
        y = img

    y = cv2.resize(y, (W, H), interpolation=cv2.INTER_CUBIC)

    if random.uniform(0, 1) < 0.5:
        y = 255 - y

    return y < np.percentile(y, random.randrange(low, high))


class ContentShuffleDetector:
    def __call__(self, img, h=None, w=None, f=None):
        H, W, C = img.shape
        if h is None:
            h = H
        if w is None:
            w = W
        if f is None:
            f = 256
        x = make_noise_disk(h, w, 1, f) * float(W - 1)
        y = make_noise_disk(h, w, 1, f) * float(H - 1)
        flow = np.concatenate([x, y], axis=2).astype(np.float32)
        return cv2.remap(img, flow, None, cv2.INTER_LINEAR)


class ColorShuffleDetector:
    def __call__(self, img):
        H, W, C = img.shape
        F = random.randint(64, 384)
        A = make_noise_disk(H, W, 3, F)
        B = make_noise_disk(H, W, 3, F)
        C = (A + B) / 2.0
        A = (C + (A - C) * 3.0).clip(0, 1)
        B = (C + (B - C) * 3.0).clip(0, 1)
        L = img.astype(np.float32) / 255.0
        Y = A * L + B * (1 - L)
        Y -= np.min(Y, axis=(0, 1), keepdims=True)
        Y /= np.maximum(np.max(Y, axis=(0, 1), keepdims=True), 1e-5)
        Y *= 255.0
        return Y.clip(0, 255).astype(np.uint8)


class GrayDetector:
    def __call__(self, img):
        eps = 1e-5
        X = img.astype(np.float32)
        r, g, b = X[:, :, 0], X[:, :, 1], X[:, :, 2]
        kr, kg, kb = [random.random() + eps for _ in range(3)]
        ks = kr + kg + kb
        kr /= ks
        kg /= ks
        kb /= ks
        Y = r * kr + g * kg + b * kb
        Y = np.stack([Y] * 3, axis=2)
        return Y.clip(0, 255).astype(np.uint8)


class DownSampleDetector:
    def __call__(self, img, level=3, k=16.0):
        h = img.astype(np.float32)
        for _ in range(level):
            h += np.random.normal(loc=0.0, scale=k, size=h.shape)
            h = cv2.pyrDown(h)
        for _ in range(level):
            h = cv2.pyrUp(h)
            h += np.random.normal(loc=0.0, scale=k, size=h.shape)
        return h.clip(0, 255).astype(np.uint8)


class Image2MaskShuffleDetector:
    def __init__(self, resolution=(640, 512)):
        self.H, self.W = resolution

    def __call__(self, img):
        m = img2mask(img, self.H, self.W)
        m *= 255.0
        return m.clip(0, 255).astype(np.uint8)
