from .config import HOME
import os
import os.path as osp
import sys
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import cv2
import numpy as np
from data.coco import get_label_map
import json
pjoin = osp.join

MIO_ROOT = osp.join(HOME, 'datasets/miotcd/')
IMAGES = 'images'
INSTANCES_SET = 'internal_cvpr2016.json'
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


def normalize_classes(cls):
    cls = cls.lower()
    dat = {'pickup truck': 'pickup_truck',
           'pickuptruck': 'pickup_truck',
           'articulated truck': 'articulated_truck',
           'articulatedtruck': 'articulated_truck',
           'non-motorized vehicle': 'non-motorized_vehicle',
           'non-motorizedvehicle': 'non-motorized_vehicle',
           'nonmotorizedvehicle': 'non-motorized_vehicle',
           'motorized vehicle': 'motorized_vehicle',
           'motorizedvehicle': 'motorized_vehicle',
           'single unit truck': 'single_unit_truck',
           'singleunittruck': 'single_unit_truck',
           'work van': 'work_van', 'suv': 'car', 'minivan': 'car', 'workvan': 'work_van'}
    return dat[cls] if cls in dat else cls


class MIOAnnotationTransform(object):
    """Transforms a MIO annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes
    """
    def __init__(self):
        self.label_map = get_label_map(osp.join(MIO_ROOT, 'miotcd_label.txt'))

    def __call__(self, target, width, height):
        """
        Args:
            target (dict): MIO target json annotation as a python dict
            height (int): height
            width (int): width
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class idx]
        """
        scale = np.array([width, height, width, height])
        res = []
        for obj in target:
            if 'xmin' in obj:
                bbox = list(map(float, [obj['xmin'], obj['ymin'], obj['xmax'] ,obj['ymax']]))
                bbox[2] += bbox[0]
                bbox[3] += bbox[1]
                label_idx = self.label_map[normalize_classes(obj['class'])] - 1
                final_box = list(np.array(bbox)/scale)
                final_box.append(label_idx)
                res += [final_box]  # [xmin, ymin, xmax, ymax, label_idx]
            else:
                print("no bbox problem!")

        return res  # [[xmin, ymin, xmax, ymax, label_idx], ... ]


class MIODetection(data.Dataset):
    """`MS MIO Detection <http://msMIO.org/dataset/#detections-challenge2016>`_ Dataset.
    Args:
        root (string): Root directory where images are downloaded to.
        set_name (string): Name of the specific set of MIO images.
        transform (callable, optional): A function/transform that augments the
                                        raw images`
        target_transform (callable, optional): A function/transform that takes
        in the target (bbox) and transforms it.
    """

    def __init__(self, root, image_set='train', transform=None,
                 target_transform=MIOAnnotationTransform(), dataset_name='MIO'):
        self.root = osp.join(root, IMAGES)
        self.mio_data = dict(self.find_data(image_set == 'train'))
        self.ids = list(self.mio_data.keys())
        self.transform = transform
        self.target_transform = target_transform
        self.name = dataset_name

    def find_data(self, is_train):
        data = json.load(open(pjoin(osp.pardir(self.root), 'json{}.json'.format(
            'train' if is_train else 'test')), 'r'))
        # v is [mio_id,items] we do not use mio_id yet.

        return [(pjoin(self.root, k+'.jpg'), v[1]) for k, v in data.items()]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target).
                   target is the object returned by ``MIO.loadAnns``.
        """
        im, gt, h, w = self.pull_item(index)
        return im, gt

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target, height, width).
                   target is the object returned by ``MIO.loadAnns``.
        """
        img_id = self.ids[index]
        mio_id, target = self.mio_data[img_id]
        path = img_id
        assert osp.exists(path), 'Image path does not exist: {}'.format(path)
        img = cv2.imread(osp.join(self.root, path))
        height, width, _ = img.shape
        if self.target_transform is not None:
            target = self.target_transform(target, width, height)
        if self.transform is not None:
            target = np.array(target)
            img, boxes, labels = self.transform(img, target[:, :4],
                                                target[:, 4])
            # to rgb
            img = img[:, :, (2, 1, 0)]

            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        return torch.from_numpy(img).permute(2, 0, 1), target, height, width

    def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            cv2 img
        '''
        img_id = self.ids[index]
        path = img_id
        return cv2.imread(osp.join(self.root, path), cv2.IMREAD_COLOR)

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
        img_id = self.ids[index]
        ann_ids = self.MIO.getAnnIds(imgIds=img_id)
        return self.MIO.loadAnns(ann_ids)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
if __name__ == '__main__':
    for x, y in MIODetection(MIO_ROOT):
        print(x.size())