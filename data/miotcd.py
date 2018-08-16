"""MIO Dataset Classes

Original author: Francisco Massa
https://github.com/fmassa/vision/blob/voc_dataset/torchvision/datasets/voc.py

Updated by: Ellis Brown, Max deGroot
"""
import json
import os
from collections import defaultdict
from random import shuffle

from .config import HOME
import os.path as osp
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np


MIO_CLASSES = [
        "articulated_truck",
        "bicycle",
        "bus",
        "car",
        "motorcycle",
        "motorized_vehicle",
        "non-motorized_vehicle",
        "pedestrian",
        "pickup_truck",
        "single_unit_truck",
        "work_van"
    ]

# note: if you used our download scripts, this should be right
MIO_ROOT = osp.join(HOME, "data/MIO/")


class MIOAnnotationTransform(object):
    """Transforms a MIO annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes

    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of MIO's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, class_to_ind=None):
        self.class_to_ind = class_to_ind or dict(
            zip(MIO_CLASSES, range(len(MIO_CLASSES))))

    def __call__(self, target, width, height):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        res = []
        for obj in target:
            name = obj['class']
            bbox = obj

            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(bbox[pt]) - 1
                # scale height or width
                cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
                bndbox.append(cur_pt)
            label_idx = self.class_to_ind[name]
            bndbox.append(label_idx)
            res += [bndbox]  # [xmin, ymin, xmax, ymax, label_ind]
            # img_id = target.find('filename').text[:-4]

        return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]


class MIODetection(data.Dataset):

    def __init__(self, root, transform=None, dataset_name='MioTCD', is_train=True):
        self.root = root
        self.is_train = is_train
        self.transform = transform
        self.name = dataset_name
        self.fnames = []
        self.targets = []
        self.num_samples = None
        self.get_absolute_path = self.get_absolute_path_train if is_train else self.get_absolute_path_test

        self.load()

    def load(self):
        list_file = 'gt_train.csv' if self.is_train else 'gt_test.csv'
        list_file = os.path.join(self.root, list_file)

        data_dict = defaultdict(list)
        with open(list_file) as f:
            for line in f:
                fields = line.strip().split(',')
                fname = fields[0]
                xmin = float(fields[2])
                ymin = float(fields[3])
                xmax = float(fields[4])
                ymax = float(fields[5])
                c = MIO_CLASSES.index(fields[1])
                target = np.array([xmin, ymin, xmax, ymax, c])
                data_dict[fname].append(target)

        for fname, target_list in data_dict.items():
            self.fnames.append(fname)
            self.targets.append(np.stack(target_list, axis=0))

    @staticmethod
    def target_transform(target, width, height):
        t = target.copy()
        t[:, 0] /= width
        t[:, 1] /= height
        t[:, 2] /= width
        t[:, 3] /= height
        return t

    def __getitem__(self, index):
        im, gt, h, w = self.pull_item(index)

        return im, gt

    def __len__(self):
        return len(self.fnames)

    def get_absolute_path_train(self, id):
        return os.path.join(self.root, 'train', id + '.jpg')

    def get_absolute_path_test(self, id):
        return os.path.join(self.root, 'test', id + '.jpg')

    def pull_item(self, index):
        fname = self.fnames[index]
        img = cv2.imread(self.get_absolute_path(fname))
        height, width, channels = img.shape
        target = self.targets[index]

        target = self.target_transform(target, width, height)
        img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])

        img = img[:, :, (2, 1, 0)]  # to rgb

        target = np.hstack((boxes, np.expand_dims(labels, axis=1)))

        return torch.from_numpy(img).permute(2, 0, 1), target, height, width
        # return torch.from_numpy(img), target, height, width

    def pull_image(self, index):
        fname = self.fnames[index]
        return cv2.imread(self.get_absolute_path(fname))

    def pull_anno(self, index):
        fname = self.fnames[index]
        gt = self.target_transform(self.targets[index], 1, 1)
        return fname, gt

    def pull_tensor(self, index):
        '''Returns the original image at an index in tensor form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            tensorized version of img, squeezed
        '''
        return torch.Tensor(self.pull_image(index)).unsqueeze_(0)
