import numpy as np
import torch


def dot_prod(a, b):
    return torch.clamp(torch.sum(a * b, -1), -1., 1.)


def dot_product_loss(y_pred, y_true):
    y_pred = (y_pred * (2 * np.pi))
    y_true = y_true * (2 * np.pi)
    a = torch.stack([torch.cos(y_pred), torch.sin(y_pred)], dim=-1)
    b = torch.stack([torch.cos(y_true), torch.sin(y_true)], dim=-1)
    res = (1.0 - dot_prod(a, b))
    return res
