import cv2
import torch
import numpy as np
from itertools import islice
from PIL import Image


def IoU(gt, pred):
    '''
    Args:
    gt: ground truth segmentation mask
    pred: predict segmentation mask
    '''
    esp = 1e-10
    intsc = torch.sum(torch.logical_and(gt, pred).float())
    union = torch.sum(torch.logical_or(gt, pred).float())
    IoU = intsc / (union + esp)
    return IoU


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def put_watermark(img, wm_encoder=None):
    if wm_encoder is not None:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDct')
        img = Image.fromarray(img[:, :, ::-1])
    return img


def plot_mask(img, masks, colors=None, alpha=0.8, indexlist=[0, 1]) -> np.ndarray:
    """Visualize segmentation mask.

    Parameters
    ----------
    img: numpy.ndarray
        Image with shape `(H, W, 3)`.
    masks: numpy.ndarray
        Binary images with shape `(N, H, W)`.
    colors: numpy.ndarray
        corlor for mask, shape `(N, 3)`.
        if None, generate random color for mask
    alpha: float, optional, default 0.5
        Transparency of plotted mask

    Returns
    -------
    numpy.ndarray
        The image plotted with segmentation masks, shape `(H, W, 3)`

    """
    H, W = masks.shape[0], masks.shape[1]
    color_list = [[255, 97, 0], [128, 42, 42], [220, 220, 220], [255, 153, 18], [56, 94, 15], [127, 255, 212], [210, 180, 140], [221, 160, 221], [255, 0, 0], [
        255, 128, 0], [255, 255, 0], [128, 255, 0], [0, 255, 0], [0, 255, 128], [0, 255, 255], [0, 128, 255], [0, 0, 255], [128, 0, 255], [255, 0, 255], [255, 0, 128]]*6
    final_color_list = [np.array([[i]*512]*512) for i in color_list]

    background = np.ones(img.shape)*255
    count = 0
    colors = final_color_list[indexlist[count]]
    for mask, color in zip(masks, colors):
        color = final_color_list[indexlist[count]]
        mask = np.stack([mask, mask, mask], -1)
        img = np.where(mask, img * (1 - alpha) + color *
                       alpha, background*0.4+img*0.6)
        count += 1
    return img.astype(np.uint8)


def color_map(N, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap


class VOCColorize(object):
    def __init__(self, n):
        self.cmap = color_map(n)

    def __call__(self, gray_image):
        size = gray_image.shape
        color_image = np.zeros((3, size[0], size[1]), dtype=np.uint8)

        for label in range(0, len(self.cmap)):
            mask = (label == gray_image)
            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        return color_image