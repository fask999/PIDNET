# ------------------------------------------------------------------------------
# Modified based on https://github.com/HRNet/HRNet-Semantic-Segmentation
# ------------------------------------------------------------------------------
import cv2
import numpy as np
import random

from torch.nn import functional as F
from torch.utils import data

y_k_size = 6
x_k_size = 6

class BaseDataset(data.Dataset):
    def __init__(self,
                 ignore_label=255,
                 base_size=2048,
                 crop_size=(512, 1024),
                 scale_factor=16,
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225],
                 bd_target='legacy', bd_mask_void=False, bd_void_radius=0):

        self.base_size = base_size
        self.crop_size = crop_size
        self.ignore_label = ignore_label

        self.mean = mean
        self.std = std
        self.scale_factor = scale_factor

        if bd_target not in ('legacy', 'valid'):
            raise ValueError("bd_target must be 'legacy' or 'valid'")
        if not isinstance(bd_void_radius, (int, np.integer)) or bd_void_radius < 0:
            raise ValueError('bd_void_radius must be a nonnegative integer')
        self.bd_target = bd_target
        self.bd_mask_void = bd_mask_void
        self.bd_void_radius = bd_void_radius

        self.files = []

    def __len__(self):
        return len(self.files)

    def input_transform(self, image, city=True):
        if city:
            image = image.astype(np.float32)[:, :, ::-1]
        else:
            image = image.astype(np.float32)
        image = image / 255.0
        image -= self.mean
        image /= self.std
        return image

    def label_transform(self, label):
        if self.ignore_label < 0:
            # Keep negative void values through OpenCV resize and padding.
            label = np.asarray(label, dtype=np.int32)
            if self.ignore_label == -1:
                label = np.where(label == 255, -1, label)
            return label
        return np.asarray(label, dtype=np.uint8)

    def pad_image(self, image, h, w, size, padvalue):
        pad_image = image.copy()
        pad_h = max(size[0] - h, 0)
        pad_w = max(size[1] - w, 0)
        if pad_h > 0 or pad_w > 0:
            pad_image = cv2.copyMakeBorder(image, 0, pad_h, 0,
                                           pad_w, cv2.BORDER_CONSTANT,
                                           value=padvalue)

        return pad_image

    def rand_crop(self, image, label, edge):
        h, w = image.shape[:-1]
        image = self.pad_image(image, h, w, self.crop_size,
                               (0.0, 0.0, 0.0))
        label = self.pad_image(label, h, w, self.crop_size,
                               (self.ignore_label,))
        if edge is not None:
            edge = self.pad_image(edge, h, w, self.crop_size, (0.0,))

        new_h, new_w = label.shape
        x = random.randint(0, new_w - self.crop_size[1])
        y = random.randint(0, new_h - self.crop_size[0])
        image = image[y:y+self.crop_size[0], x:x+self.crop_size[1]]
        label = label[y:y+self.crop_size[0], x:x+self.crop_size[1]]
        if edge is not None:
            edge = edge[y:y+self.crop_size[0], x:x+self.crop_size[1]]

        return image, label, edge

    def multi_scale_aug(self, image, label=None, edge=None,
                        rand_scale=1, rand_crop=True):
        long_size = int(self.base_size * rand_scale + 0.5)
        h, w = image.shape[:2]
        if h > w:
            new_h = long_size
            new_w = int(w * long_size / h + 0.5)
        else:
            new_w = long_size
            new_h = int(h * long_size / w + 0.5)

        image = cv2.resize(image, (new_w, new_h),
                           interpolation=cv2.INTER_LINEAR)
        if label is not None:
            label = cv2.resize(label, (new_w, new_h),
                               interpolation=cv2.INTER_NEAREST)
            if edge is not None:
                edge = cv2.resize(edge, (new_w, new_h),
                                   interpolation=cv2.INTER_NEAREST)
        else:
            return image

        if rand_crop:
            image, label, edge = self.rand_crop(image, label, edge)

        return image, label, edge


    @staticmethod
    def _legacy_edge(label, edge_pad, edge_size):
        edge = cv2.Canny(label.astype(np.uint8, copy=False), 0.1, 0.2)
        kernel = np.ones((edge_size, edge_size), np.uint8)
        if edge_pad:
            edge[:y_k_size] = 0
            edge[-y_k_size:] = 0
            edge[:, :x_k_size] = 0
            edge[:, -x_k_size:] = 0
        return (cv2.dilate(edge, kernel, iterations=1) > 50) * 1.0

    @staticmethod
    def _valid_edge(label, valid, edge_size):
        if not isinstance(edge_size, (int, np.integer)) or edge_size < 1 or edge_size % 2 == 0:
            raise ValueError('valid boundary width must be a positive odd integer')

        edge = np.zeros(label.shape, dtype=np.uint8)
        vertical = valid[:-1, :] & valid[1:, :] & (label[:-1, :] != label[1:, :])
        horizontal = valid[:, :-1] & valid[:, 1:] & (label[:, :-1] != label[:, 1:])
        edge[:-1, :] |= vertical
        edge[1:, :] |= vertical
        edge[:, :-1] |= horizontal
        edge[:, 1:] |= horizontal

        # Grow within the valid domain one pixel at a time, so a wider kernel
        # cannot jump a void or padding band.
        kernel = np.ones((3, 3), dtype=np.uint8)
        for _ in range(edge_size // 2):
            edge = cv2.dilate(edge, kernel)
            edge[~valid] = 0
        return edge.astype(np.float64)

    def gen_sample(self, image, label,
                   multi_scale=True, is_flip=True, edge_pad=True, edge_size=4, city=True):
        label = self.label_transform(label)
        edge = self._legacy_edge(label, edge_pad, edge_size) if self.bd_target == 'legacy' else None

        if multi_scale:
            rand_scale = 0.5 + random.randint(0, self.scale_factor) / 10.0
            image, label, edge = self.multi_scale_aug(image, label, edge,
                                                rand_scale=rand_scale)

        image = self.input_transform(image, city=city)

        image = image.transpose((2, 0, 1))

        if is_flip:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]
            label = label[:, ::flip]
            if edge is not None:
                edge = edge[:, ::flip]

        if self.bd_target == 'valid' or self.bd_mask_void:
            valid = label != self.ignore_label
            if self.bd_target == 'valid':
                edge = self._valid_edge(label, valid, edge_size)
            if self.bd_mask_void:
                invalid = ~valid
                if self.bd_void_radius:
                    width = 2 * self.bd_void_radius + 1
                    kernel = np.ones((width, width), dtype=np.uint8)
                    invalid = cv2.dilate(invalid.astype(np.uint8), kernel).astype(bool)
                edge[invalid] = -1.0

        return image, label, edge


    def inference(self, config, model, image):
        size = image.size()
        pred = model(image)

        if config.MODEL.NUM_OUTPUTS > 1:
            pred = pred[config.TEST.OUTPUT_INDEX]
        
        
        pred = F.interpolate(
            input=pred, size=size[-2:],
            mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
        )
        
        
        return pred.exp()

