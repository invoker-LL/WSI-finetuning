import torch
import numpy as np
from visdom import Visdom
# import wandb
import os
import torch.utils.data as data
from PIL import Image
from io import BytesIO
import cv2
from wand.image import Image as WandImage
from wand.api import library as wandlibrary
from scipy.ndimage import zoom as scizoom
from skimage import color
from difflib import SequenceMatcher
import torch.nn as nn
import torch.nn.functional as F
import sklearn.covariance
import torchvision.datasets as dset
import torchvision.transforms as trn
from PIL import ImageFilter
import random

def metrics(prediction, target):

    prediction_binary = torch.argmax(prediction, dim=1)
    N = target.numel()

    # True positives, true negative, false positives, false negatives calculation
    tp = torch.nonzero(prediction_binary * target).shape[0]
    tn = torch.nonzero((1 - prediction_binary) * (1 - target)).shape[0]
    fp = torch.nonzero(prediction_binary * (1 - target)).shape[0]
    fn = torch.nonzero((1 - prediction_binary) * target).shape[0]

    # Metrics
    accuracy = (tp + tn) / N
    precision = (tp + 1e-4) / (tp + fp + 1e-4)
    recall = (tp + 1e-4) / (tp + fn + 1e-4)
    specificity = (tn + 1e-4) / (tn + fp + 1e-4)
    f1 = (2 * precision * recall + 1e-4) / (precision + recall + 1e-4)

    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1, 'specificity': specificity}




def clipped_zoom(img, zoom_factor):
    h = img.shape[0]
    # ceil crop height(= crop width)
    ch = int(np.ceil(h / zoom_factor))

    top = (h - ch) // 2
    img = scizoom(img[top:top + ch, top:top + ch], (zoom_factor, zoom_factor, 1), order=1)
    # trim off any extra pixels
    trim_top = (img.shape[0] - h) // 2

    return img[trim_top:trim_top + h, trim_top:trim_top + h]




def disk(radius, alias_blur=0.1, dtype=np.float32):
    if radius <= 8:
        L = np.arange(-8, 8 + 1)
        ksize = (3, 3)
    else:
        L = np.arange(-radius, radius + 1)
        ksize = (5, 5)
    X, Y = np.meshgrid(L, L)
    aliased_disk = np.array((X ** 2 + Y ** 2) <= radius ** 2, dtype=dtype)
    aliased_disk /= np.sum(aliased_disk)

    # supersample disk to antialias
    return cv2.GaussianBlur(aliased_disk, ksize=ksize, sigmaX=alias_blur)




def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)

def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

# Extend wand.image.Image class to include method signature
class MotionImage(WandImage):
    def motion_blur(self, radius=0.0, sigma=0.0, angle=0.0):
        wandlibrary.MagickMotionBlurImage(self.wand, radius, sigma, angle)

def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

def make_dataset(dir, class_to_idx, ratio=1.0):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            leng = int(len(fnames) * ratio)
            for fname in sorted(fnames)[:leng]:
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']

def is_image_file(filename):
    """Checks if a file is an image.
    Args:
        filename (string): path to a file
    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)


class PartialImageFolder(data.Dataset):
    def __init__(self, root, ratio=1.0, transform=None, target_transform=None,
                 loader=default_loader):
        classes, class_to_idx = find_classes(root)
        imgs = make_dataset(root, class_to_idx, ratio)
        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                                                                             "Supported image extensions are: " + ",".join(
                IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.idx_to_class = {v: k for k, v in class_to_idx.items()}
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)


class DistortImageFolder(data.Dataset):
    def __init__(self, root, method, severity, transform=None, target_transform=None,
                 loader=default_loader):
        classes, class_to_idx = find_classes(root)
        imgs = make_dataset(root, class_to_idx)
        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                                                                             "Supported image extensions are: " + ",".join(
                IMG_EXTENSIONS)))

        self.root = root
        self.distor = Distortions()
        self.method = getattr(self.distor, method)
        self.severity = severity
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.idx_to_class = {v: k for k, v in class_to_idx.items()}
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        img = self.method(img, self.severity)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)


class Distortions(object):
    def __init__(self, ):
        # self.blots = np.load('blots.npy', allow_pickle=True)
        # self.bubbles = np.load('bubbles.npy', allow_pickle=True)
        pass

    def pixelate(self, x, severity=1):
        c = [0.6, 0.5, 0.4, 0.3, 0.25][severity - 1]
        dims = x.size

        x = x.resize((int(dims[0] * c), int(dims[1] * c)), Image.BOX)
        x = x.resize(dims, Image.BOX)

        return x

    def jpeg_compression(self, x, severity=1):
        c = [25, 18, 15, 10, 7][severity - 1]

        output = BytesIO()
        x.save(output, 'JPEG', quality=c)
        x = Image.open(output)

        return x

    def marking_blur(self, x, severity=1):
        x = np.array(x)
        blot = dict(self.blots[0]['96'][severity-1])
        blur = blot["blur"]
        binary = blot["mask"]
        rand_x, rand_y = blot["positions"][0]
        blur_h = blot["blur_h"]
        blur_w = blot["blur_w"]
        blur = x[rand_y: rand_y + blur_h, rand_x:rand_x + blur_w] * (1 - binary) + blur
        x[rand_y: rand_y + blur_h, rand_x:rand_x + blur_w] = blur

        return x

    def bubble_blur(self, x, severity=1):

        ratios = [0.1, 0.3, 0.5, 0.7, 0.9]
        ratio = ratios[severity-1]
        x = np.array(x)
        sample = dict(self.bubbles[0]['96'])
        bubble = sample["bubble"]
        mask = sample["mask"]
        x = x * mask * (1-ratio) + bubble * mask * ratio + x * (1 - mask)

        return np.clip(x, 0, 255).astype(np.uint8)

    def defocus_blur(self, x, severity=1):

        if x.size == (224, 224):
            c = [(3, 0.1), (4, 0.5), (6, 0.5), (8, 0.5), (10, 0.5)][severity - 1]
        elif x.size == (96, 96):
            c = [(3, 0.05), (4, 0.3), (5, 0.3), (6, 0.3), (7, 0.3)][severity - 1]

        x = np.array(x) / 255.
        kernel = disk(radius=c[0], alias_blur=c[1])

        channels = []
        for d in range(3):
            channels.append(cv2.filter2D(x[:, :, d], -1, kernel))
        channels = np.array(channels).transpose((1, 2, 0))  # 3x224x224 -> 224x224x3

        return (np.clip(channels, 0, 1) * 255).astype(np.uint8)

    def motion_blur(self, x, severity=1):
        dims = x.size
        if x.size == (224, 224):
            c = [(10, 3), (15, 5), (15, 8), (15, 12), (20, 15)][severity - 1]
        elif x.size == (96, 96):
            c = [(5, 1), (5, 2), (5, 3), (5, 4), (6, 5)][severity - 1]

        output = BytesIO()
        x.save(output, format='PNG')
        x = MotionImage(blob=output.getvalue())

        x.motion_blur(radius=c[0], sigma=c[1], angle=np.random.uniform(-45, 45))

        x = cv2.imdecode(np.fromstring(x.make_blob(), np.uint8),
                         cv2.IMREAD_UNCHANGED)

        if x.shape != dims:
            return np.clip(x[..., [2, 1, 0]], 0, 255)  # BGR to RGB
        else:  # greyscale to RGB
            return np.clip(np.array([x, x, x]).transpose((1, 2, 0)), 0, 255)

    def zoom_blur(self, x, severity=1):
        c = [np.arange(1, 1.11, 0.01),
             np.arange(1, 1.16, 0.01),
             np.arange(1, 1.21, 0.02),
             np.arange(1, 1.26, 0.02),
             np.arange(1, 1.31, 0.03)][severity - 1]

        x = (np.array(x) / 255.).astype(np.float32)
        out = np.zeros_like(x)
        for zoom_factor in c:
            out += clipped_zoom(x, zoom_factor)

        x = (x + out) / (len(c) + 1)
        return (np.clip(x, 0, 1) * 255).astype(np.uint8)

    def brightness(self, x, severity=1):
        c = [.05, .1, .15, .2, .25][severity - 1]

        x = np.array(x) / 255.
        x = color.rgb2hsv(x)
        x[:, :, 2] = np.clip(x[:, :, 2] + c, 0, 1)
        x = color.hsv2rgb(x)

        return (np.clip(x, 0, 1) * 255).astype(np.uint8)

    def saturate(self, x, severity=1):
        # c = [(0.3, 0), (0.1, 0), (2, 0), (5, 0.1), (20, 0.2)][severity - 1]
        # c = [(0.3, 0), (0.1, 0), (2, 0), (5, 0.1), (20, 0.2)][severity - 1]
        c = [.05, .1, .15, .2, .25][severity - 1]

        x = np.array(x) / 255.
        x = color.rgb2hsv(x)
        x[:, :, 1] = np.clip(x[:, :, 1] + c, 0, 1)
        x = color.hsv2rgb(x)

        return (np.clip(x, 0, 1) * 255).astype(np.uint8)

    def hue(self, x, severity=1):
        # this implementation is unreliable and needs debug
        c = [.02, .04, .06, .08, .1][severity - 1]

        x = np.array(x) / 255.
        x = color.rgb2hsv(x)
        x[:,:,0] = np.where(x[:, :, 0] + c > 1.0, x[:, :, 0] + c -1, x[:, :, 0] + c)
        x = color.hsv2rgb(x)

        return (np.clip(x, 0, 1) * 255).astype(np.uint8)


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

__all__ = ["Registry"]


class Registry:
    """A registry providing name -> object mapping, to support
    custom modules.

    To create a registry (e.g. a backbone registry):

    .. code-block:: python

        BACKBONE_REGISTRY = Registry('BACKBONE')

    To register an object:

    .. code-block:: python

        @BACKBONE_REGISTRY.register()
        class MyBackbone(nn.Module):
            ...

    Or:

    .. code-block:: python

        BACKBONE_REGISTRY.register(MyBackbone)
    """

    def __init__(self, name):
        self._name = name
        self._obj_map = dict()

    def _do_register(self, name, obj, force=False):
        if name in self._obj_map and not force:
            raise KeyError(
                'An object named "{}" was already '
                'registered in "{}" registry'.format(name, self._name)
            )

        self._obj_map[name] = obj

    def register(self, obj=None, force=False):
        if obj is None:
            # Used as a decorator
            def wrapper(fn_or_class):
                name = fn_or_class.__name__
                self._do_register(name, fn_or_class, force=force)
                return fn_or_class

            return wrapper

        # Used as a function call
        name = obj.__name__
        self._do_register(name, obj, force=force)

    def get(self, name):
        if name not in self._obj_map:
            raise KeyError(
                'Object name "{}" does not exist '
                'in "{}" registry'.format(name, self._name)
            )

        return self._obj_map[name]

    def registered_names(self):
        return list(self._obj_map.keys())

def init_network_weights(model, init_type="normal", gain=0.02):

    def _init_func(m):
        classname = m.__class__.__name__

        if hasattr(m, "weight") and (
            classname.find("Conv") != -1 or classname.find("Linear") != -1
        ):
            if init_type == "normal":
                nn.init.normal_(m.weight.data, 0.0, gain)
            elif init_type == "xavier":
                nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == "kaiming":
                nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                nn.init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError(
                    "initialization method {} is not implemented".
                    format(init_type)
                )
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)

        elif classname.find("BatchNorm") != -1:
            nn.init.constant_(m.weight.data, 1.0)
            nn.init.constant_(m.bias.data, 0.0)

        elif classname.find("InstanceNorm") != -1:
            if m.weight is not None and m.bias is not None:
                nn.init.constant_(m.weight.data, 1.0)
                nn.init.constant_(m.bias.data, 0.0)

    model.apply(_init_func)

def get_most_similar_str_to_a_from_b(a, b):
    """Return the most similar string to a in b.

    Args:
        a (str): probe string.
        b (list): a list of candidate strings.
    """
    highest_sim = 0
    chosen = None
    for candidate in b:
        sim = SequenceMatcher(None, a, candidate).ratio()
        if sim >= highest_sim:
            highest_sim = sim
            chosen = candidate
    return chosen

def check_availability(requested, available):
    """Check if an element is available in a list.

    Args:
        requested (str): probe string.
        available (list): a list of available strings.
    """
    if requested not in available:
        psb_ans = get_most_similar_str_to_a_from_b(requested, available)
        raise ValueError(
            "The requested one is expected "
            "to belong to {}, but got [{}] "
            "(do you mean [{}]?)".format(available, requested, psb_ans)
        )

def print_log(print_string, log):
    print("{}".format(print_string))
    log.write('{}\n'.format(print_string))
    log.flush()


def sort_metric(conf_array):
    (num, c) = conf_array.shape
    total = c * (c-1) / 2
    avg_change = 0
    for i in range(num):
        swapped = True
        cur_conf = conf_array[i]
        changes = 0
        last = c
        while swapped:
            swapped = False
            for j in range(1, last):
                if cur_conf[j - 1] > cur_conf[j]:
                    cur_conf[j], cur_conf[j - 1] = cur_conf[j - 1], cur_conf[j]  # Swap
                    changes += 1
                    swapped = True
                    last = j
        avg_change = (avg_change * i + 1.0 - changes / total) / (i+1)
    return avg_change