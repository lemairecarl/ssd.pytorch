import csv
import json
import os
from collections import defaultdict

import cv2
import numpy as np
from scipy import stats
import torch
from tqdm import tqdm

from layers.box_utils import jaccard
import subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('root', help='Dataset root')
parser.add_argument('csv', help='CSV path')
args = parser.parse_args()
pjoin = os.path.join

gt_test = pjoin(args.root, 'gt_test.csv')
print(subprocess.run(['python', 'localization_evaluation.py', gt_test, args.csv],
                     stdout=subprocess.PIPE,
                     stderr=subprocess.PIPE).stdout.decode('utf-8'))


gt = json.load(open(pjoin(args.root, 'jsontest.json'), 'r'))
pred = defaultdict(list)
for id, label, conf, xmin, ymin, xmax, ymax, angle, parked in csv.reader(open(args.csv, 'r')):
    d = {'angle': float(angle),
         'parked': float(parked),
         'class': label,
         'conf': float(conf),
         'xmin': float(xmin), 'xmax': float(xmax), 'ymin': float(ymin), 'ymax': float(ymax)}
    pred[id].append(d)


def show():
    gt_clr = (255, 0, 0)
    pred_clr = (0, 255, 0)
    for k, [_, vals] in gt.items():
        img = cv2.imread(pjoin(args.root, 'images', k + '.jpg'))
        for b in vals:
            draw_rect(img, b, gt_clr)
        for b in pred[k]:
            if b['conf'] < 0.1:
                continue
            draw_rect(img, b, pred_clr)
        cv2.imshow('', img)
        cv2.waitKey(1000)


def iou(b1, b2):
    b1_arr = np.array([b1['xmin'], b1['ymin'], b1['xmax'], b1['ymax']]).reshape([1, 4]).astype(np.float)
    b2_arr = np.array([b2['xmin'], b2['ymin'], b2['xmax'], b2['ymax']]).reshape([1, 4]).astype(np.float)
    return jaccard(torch.from_numpy(b1_arr), torch.from_numpy(b2_arr)).numpy()[0].item()


def find_best_iou(b1, vals, keep_parked=False):
    best = max([(b2, iou(b1, b2)) for b2 in vals], key=lambda k: k[1])
    if (not keep_parked and float(best[0]['mag']) < 2.) or best[1] < 0.75:
        return None
    else:
        return best[0]


def draw_rect(img, b, clr):
    cv2.rectangle(img, (int(b['xmin']), int(b['ymin'])), (int(b['xmax']), int(b['ymax'])), clr, 1)
    if 'parked' in b:
        prk_clr = (0, 255, 0) if b['parked'] < 0.5 else (0, 0, 255)
    else:
        prk_clr = (0, 255, 255)
    cx, cy = map(int, [np.mean([b['xmin'], b['xmax']]), np.mean([b['ymin'], b['ymax']])])
    cx2 = int(cx + 20 * np.cos(float(b['angle'])))
    cy2 = int(cy + 20 * np.sin(float(b['angle'])))
    cv2.arrowedLine(img, (cx, cy), (cx2, cy2),
                    prk_clr, 1, tipLength=1)


def cosine_distance(b1, b2):
    a, b = map(float,(b1['angle'], b2['angle']))
    av = np.array([np.cos(a), np.sin(a)])
    bv = np.array([np.cos(b), np.sin(b)])
    dot = np.dot(av, bv)
    return np.arccos(np.clip(dot, -1, 1))



def score():
    score = []
    for k, boxes in tqdm(pred.items()):
        [_, vals] = gt[k]
        for b in boxes:
            if float(b['conf']) < 0.5:
                continue
            bbox = find_best_iou(b, vals, keep_parked=False)
            if bbox is None:
                continue
            cos_dist = cosine_distance(b, bbox)
            score.append(cos_dist)
    return score
#show()
print(stats.describe(score()))
