"""MIO Dataset Classes

Original author: Francisco Massa
https://github.com/fmassa/vision/blob/voc_dataset/torchvision/datasets/voc.py

Updated by: Ellis Brown, Max deGroot
"""
import json
import os.path as osp
from random import shuffle, seed
import warnings

import cv2
import h5py
import numpy as np
import torch
import torch.utils.data as data

from data.config import HOME
from utils import SSDAugmentation

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
            angle = float(obj['angle']) / (2 * np.pi)
            parked = float(float(obj['mag']) < 1.)
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
            bndbox.append(angle)
            bndbox.append(parked)
            res += [bndbox]  # [xmin, ymin, xmax, ymax, label_ind, angle, parked]
            # img_id = target.find('filename').text[:-4]

        return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]


class MIODetection(data.Dataset):
    """MIO Detection Dataset Object

    input is image, target is annotation

    Arguments:
        root (string): filepath to MIOdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'MIO2007')
    """

    def __init__(self, root,
                 transform=None, target_transform=MIOAnnotationTransform(),
                 dataset_name='mioTCD', is_train=True):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.name = dataset_name
        self._annopath = osp.join(self.root, 'json{}.json'.format('train' if is_train else 'test'))
        self._imgpath = osp.join(self.root, 'images', '%s.jpg')
        self.get_h5pyfile = lambda: h5py.File(osp.join(self.root, 'apriori2.h5'), 'r')
        self.is_train = is_train
        self.ids = list()
        data = json.load(open(self._annopath))
        items = list(data.items())
        seed(1337)
        shuffle(items)
        with self.get_h5pyfile() as f:
            for k, [[_, video_id], vals] in items:
                if video_id in f:
                    # 2-3 per file
                    self.ids.append((k, (video_id, vals)))

    def __getitem__(self, index):
        im, gt, odf, h, w = self.pull_item(index)

        return im, gt, odf

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        img_id, [video_id, target] = self.ids[index]
        img = cv2.imread(self._imgpath % img_id)
        height, width, channels = img.shape
        with self.get_h5pyfile() as f:
            odf = f[video_id].value  # Remove the uniform dist

        p = [.5] + [.5 / odf.shape[0]] * odf.shape[0]
        to_take = np.random.choice(np.arange(0, odf.shape[0] + 1), p=p)
        to_take = to_take if self.is_train else odf.shape[0]
        if to_take == 0:
            # Initial uniform odf
            odf = np.ones([19, 19, odf.shape[-1]]) / odf.shape[-1]
        else:
            # Remove all but one uniform
            odf = odf[np.random.choice(np.arange(0, to_take), to_take, replace=False)].sum(0)
            odf = self.normalize_odf(odf)
        # Add noise
        if self.is_train:
            # 10% noise
            odf += np.random.normal(0, 0.05, size=[19, 19, odf.shape[-1]])
            odf = self.normalize_odf(odf)

        if self.target_transform is not None:
            target = self.target_transform(target, width, height)

        if self.transform is not None:
            target = np.array(target)
            img, boxes, labels, odf = self.transform(img,  target[:, :4], target[:, 4:], odf=odf)
            # to rgb
            img = img[:, :, (2, 1, 0)]
            # img = img.transpose(2, 0, 1)
            target = np.hstack((boxes, labels))
            odf = self.normalize_odf(odf)
        return (torch.from_numpy(img).permute(2, 0, 1),
                target,
                torch.from_numpy(np.copy(odf).astype(np.float32)).permute(2, 0, 1),
                height,
                width)

    def normalize_odf(self, odf):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            odf = odf / odf.sum(-1, keepdims=True)
            odf[np.isnan(odf)] = 0
        return odf

    def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            PIL img
        '''
        img_id = self.ids[index][0]
        return cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR)

    def pull_anno(self, index):
        '''Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        img_id, [_, anno] = self.ids[index][1]
        gt = self.target_transform(anno, 1, 1)
        return img_id, gt

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

def draw_odf(odf, img):
    width, height = 608, 608
    odf = odf.numpy()
    fx = width / 19
    fy = height / 19
    box = list(range(19))
    angles = np.linspace(0, 1.,11)
    cst = 50
    for i,j in product(box,box):
        cx = int((i + 0.5) * fx)
        cy = int((j + 0.5) * fy)
        o = odf[:,j, i]
        i1 = np.argmax(o)
        cx2 = cx + np.cos(angles[i1] * 2 * np.pi) * cst * o[i1]
        cy2 = cy + np.sin(angles[i1] * 2 * np.pi) * cst * o[i1]
        cx2, cy2 = map(int, [cx2, cy2])
        cv2.arrowedLine(img, (cx,cy), (cx2,cy2), (0, 255, 0),tipLength=0.3)
    cv2.imshow('lol', img)
    cv2.waitKey(10000)


if __name__ == '__main__':
    from itertools import product
    MEANS = (104, 117, 123)
    d = MIODetection('/data/mio_tcd_seg', transform=SSDAugmentation(300,
                                                                    MEANS), is_train=False)
    d1 = MIODetection('/data/mio_tcd_seg', transform=None, is_train=False)
    for idx in range(100):
        # img = d.pull_image(idx)
        img, _, odf, height, width = d.pull_item(idx)
        img = img.permute(1, 2, 0).numpy()
        img = cv2.resize(img, (608,608))

        draw_odf(odf, img.copy())
        img, _, odf, height, width = d1.pull_item(idx)
        img = img.permute(1, 2, 0).numpy()
        img = cv2.resize(img, (608, 608))
        draw_odf(odf, img.copy())



