from __future__ import print_function

import argparse
import cv2
import os

import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from tqdm import tqdm

from data import BaseTransform, VOC_CLASSES, MIO_CLASSES
from data import MIO_CLASSES as labelmap, MIO_ROOT, MIOAnnotationTransform, MIODetection
from ssd import build_ssd

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
parser.add_argument('--trained_model', default='ssd300_COCO_90000.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='.', type=str,
                    help='Dir to save results')
parser.add_argument('--visual_threshold', default=0.5, type=float,
                    help='Final confidence threshold')
parser.add_argument('--cuda', action='store_true',
                    help='Use cuda to train model')
parser.add_argument('--root', default='/home/fred/datasets/miotcd', help='Location of VOC root directory')
parser.add_argument('-f', default=None, type=str, help="Dummy arg so we can load in Jupyter Notebooks")
args = parser.parse_args()

if args.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)


def test_net(save_folder, net, cuda, testset, transform, thresh):
    num_images = len(testset)
    results = []
    for i in tqdm(range(num_images)):
        print('Testing image {:d}/{:d}....'.format(i + 1, num_images))
        img = testset.pull_image(i)
        img_id = testset.ids[i][0]
        x = torch.from_numpy(transform(img)[0][..., (2, 1, 0)]).permute(2, 0, 1)
        x = Variable(x.unsqueeze(0))

        if cuda:
            x = x.cuda()

        y = net(x)  # forward pass
        detections = y.data
        # scale each detection back up to the image
        scale = torch.Tensor([img.shape[1], img.shape[0],
                              img.shape[1], img.shape[0]])

        for i in range(detections.size(1)):
            j = 0
            while detections[0, i, j, 0] >= thresh:
                score = detections[0, i, j, 0]
                label_name = labelmap[i - 1]
                pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
                coords = (pt[0], pt[1], pt[2], pt[3])
                coords_int = tuple(map(int, coords))
                results.append([img_id, score, label_name, *coords])
                cv2.rectangle(img,(coords_int[0], coords_int[1]), (coords_int[2], coords_int[3]),(255, 0, 0))
                j += 1
        cv2.imshow('lol', img)
        cv2.waitKey(500)

    with open('machin.csv') as f:
        f.writelines([','.join(map(str, k)) for k in results])


def test_voc():
    # load net
    num_classes = len(MIO_CLASSES) + 1  # +1 background
    net = build_ssd('test', 300, num_classes)  # initialize SSD
    net.load_state_dict(torch.load(args.trained_model, map_location='cpu' if not args.cuda else None))
    net.eval()
    print('Finished loading model!')
    # load data
    testset = MIODetection(args.root, None, MIOAnnotationTransform(), is_train=False)
    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True
    # evaluation
    test_net(args.save_folder, net, args.cuda, testset,
             BaseTransform(net.size, (104, 117, 123)),
             thresh=args.visual_threshold)


if __name__ == '__main__':
    test_voc()
