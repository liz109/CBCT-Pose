import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F


# Function to convert a nested dictionary to a nested argparse.Namespace
def dict_to_namespace(d):
    namespace = argparse.Namespace()
    for key, value in d.items():
        if isinstance(value, dict):
            setattr(namespace, key, dict_to_namespace(value))
        else:
            setattr(namespace, key, value)
    return namespace


def rgb_to_gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


def rescale(image, min, max):
    rescale_image = min + (image-image.min()) * (max-min)/(image.max()-image.min())
    return rescale_image


def denormalize(image, norm_range_min, norm_range_max):
    image = image * (norm_range_max - norm_range_min) + norm_range_min
    return image


def normalize(image, norm_range_min, norm_range_max):
    # "Normalize range to [0, 1]"
    image = (image - norm_range_min) / (norm_range_max - norm_range_min)
    return image


def trunc(mat, trunc_min=0.0, trunc_max=1.0):
    mat[mat <= trunc_min] = trunc_min
    mat[mat >= trunc_max] = trunc_max
    return mat


