import collections.abc as abc
import copy
import datetime
import gc
import glob
import hashlib
import inspect
import io
import itertools
import json
import seaborn as sns
import logging
import math
import os
import pickle
import platform
import random
import re
import secrets
import shutil
import sys
import tempfile
import textwrap
import time
import timeit
import traceback
import typing
from collections import OrderedDict, defaultdict, namedtuple
from dataclasses import dataclass
from functools import cached_property, partial
from inspect import signature
from itertools import chain, count
from pathlib import Path
from pprint import pprint
from typing import Annotated

import cv2
import einops
import ipyplot
import IPython
import IPython.display
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import p_tqdm
import plotly.express as px
import plotly.graph_objects as go
import seaborn
import shapely
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms.v2 as transforms
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from IPython import get_ipython
from IPython.display import display
from numba import njit
from PIL import Image, ImageOps
from plotly.subplots import make_subplots
from rich import print as print_tmp
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from shapely import ops, union, union_all
from shapely.geometry import LineString, MultiLineString, MultiPoint, Point, Polygon
from sklearn.metrics import accuracy_score, classification_report
from tabulate import tabulate
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
from torchmetrics import Accuracy
from torchmetrics.classification import MulticlassAccuracy
from torchvision.ops.boxes import complete_box_iou
from torchvision.utils import make_grid
from tqdm import tqdm

import wandb


class StopExecution(Exception):
    def _render_traceback_(self):
        return []


def exit():
    raise StopExecution


class HiddenPrints:
    def __init__(self, disable=False):
        self.disable = disable

    def __enter__(self):
        if not self.disable:
            self._original_stdout = sys.stdout
            sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.disable:
            sys.stdout.close()
            sys.stdout = self._original_stdout


class Timer:
    def __init__(self):
        self.start = time.time()

    def elapsed(self):
        return time.time() - self.start

    def reset(self):
        self.start = time.time()


def benchmark(func, times=1000000):
    return timeit.Timer(func).timeit(number=times)


@njit
def norm1(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return abs(x1 - x2) + abs(y1 - y2)


def norm1_s(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return abs(x1 - x2) + abs(y1 - y2)


@njit
def norm2(a, b):
    x1, y1 = a
    x2, y2 = b
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


@njit
def snorm2_njit(a, b):
    x1, y1 = a
    x2, y2 = b
    return (x1 - x2) ** 2 + (y1 - y2) ** 2


def snorm2(a, b):
    x1, y1 = a
    x2, y2 = b
    return (x1 - x2) ** 2 + (y1 - y2) ** 2


def column(matrix, i):
    return [row[i] for row in matrix]


DEBUG = True


def is_notebook() -> bool:
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


def print(*args, **kwargs):
    if not DEBUG:
        return
    info = traceback.format_stack()[-2]
    end = info.index(",", info.index(",") + 1)
    line_number = traceback.format_stack()[-2][7:end]
    if not is_notebook():
        # print_tmp(*args, f"{line_number}", **kwargs)
        print_tmp(*args, **kwargs)
    else:
        print_tmp(*args, **kwargs)


def color_map(n):
    from pypalettes import get_hex, get_rgb, load_cmap

    return get_rgb(["Signac", "Antique"])[n]


# for x in mpl.colormaps:
#     print(x)
#     try:
#         print(len(mpl.colormaps[x].colors))
#     except:
#         pass
# color_map(0)
# print(mpl.colormaps)
# exit()


def shapely_to_numpy(shapely_obj):
    if isinstance(shapely_obj, Point):
        return np.array(shapely_obj.coords)
    else:
        raise ValueError("Not supported type")


def plotly_to_array(fig):
    image_bytes = fig.to_image(format="jpg")
    return np.asarray(Image.open(io.BytesIO(image_bytes)))


def seaborn_to_array(fig):
    return np.asarray(fig.get_figure().canvas.buffer_rgba())


def visualize_attentions(maps, has_attention=False):
    map_num = len(maps[0]) - 1
    fig = make_subplots(rows=1, cols=map_num + 1, horizontal_spacing=0)
    if has_attention:
        for i in range(len(maps)):
            for j in range(1, map_num + 1):
                fig.add_heatmap(
                    z=maps[i][j],
                    visible=False,
                    row=1,
                    col=j,
                    showscale=True if j == 1 else False,
                )
            fig.add_image(z=maps[i][0], visible=False, row=1, col=map_num + 1)
    else:
        for i in range(len(maps)):
            for j in range(map_num + 1):
                fig.add_image(z=maps[i][j], visible=False, row=1, col=j + 1)
    for i in range(map_num + 1):
        fig.data[i].visible = True
    steps = []
    for i in range(0, len(fig.data), map_num + 1):
        step = dict(
            method="update",
            args=[{"visible": [False] * len(fig.data)}],
        )
        for j in range(map_num + 1):
            step["args"][0]["visible"][i + j] = True
        steps.append(step)
    sliders = [dict(active=10, currentvalue={"prefix": "index: "}, pad={"t": 50}, steps=steps)]
    for i in range(map_num + 1):
        fig.update_xaxes(showgrid=False, zeroline=False, visible=False)
    for i in range(map_num + 1):
        fig.update_yaxes(
            row=1,
            col=i + 1,
            autorange="reversed",
            scaleanchor=f"x{i+1}",
            scaleratio=1,
            showgrid=False,
            zeroline=False,
            visible=False,
        )
    fig.update_layout(
        sliders=sliders,
        plot_bgcolor="white",
        # autosize = False
    )
    fig.show()


def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func

    return decorate


def flatten_list(lst):
    flattened_list = []
    for item in lst:
        if isinstance(item, list) or isinstance(item, tuple):
            flattened_list.extend(flatten_list(item))
        elif isinstance(item, torch.Tensor) or isinstance(item, np.ndarray):
            flattened_list.append(item)
    return flattened_list


def plot_images(images, img_width=None, max_images=5, parallel=False, parallel_size=5):
    if not isinstance(images, abc.Sequence) and (
        isinstance(images, np.ndarray) and len(images.shape) == 3 or len(images.shape) == 2
    ):
        images = [images]
    if not is_notebook():
        for image in images:
            with tempfile.NamedTemporaryFile(suffix=".png") as f:
                cv2.imwrite(f.name, image)
                # os.system(
                #     f"convert {f.name} -resize {img_width if img_width else 200} -alpha off sixel:-"
                # )
                print()
                os.system(f"img2sixel -w{img_width if img_width else 200} {f.name}")
                print()
        return
    images = images[:max_images]
    L = len(images)
    images = flatten_list(images)
    for i in range(len(images)):
        if isinstance(images[i], torch.Tensor):
            images[i] = transforms.ToPILImage()(images[i])
            images[i] = np.array(images[i])
            scale = images[i].max() > 1 and images[i].dtype == torch.float32
            images[i] = np.array(transforms.ToPILImage()(images[i]))
            if scale:
                images[i] = 255 - images[i]

    cols = len(images) // L
    if len(images) == 1:
        images = images[0]
        height = images.shape[0] if img_width == -1 else img_width if img_width else 200
        display(ImageOps.contain(Image.fromarray(images), (height, height)))
    else:
        if not parallel:
            for i in range(0, len(images), cols):
                height = images[i].shape[0] if img_width == -1 else img_width if img_width else 200
                ipyplot.plot_images(
                    images[i : i + cols],
                    img_width=height,
                )
        else:
            ipyplot.plot_images(
                images,
                max_images=parallel_size,
                img_width=img_width if img_width else 200,
            )


class ThresholdTransform(object):
    def __init__(self, thr_255):
        self.thr = thr_255

    def __call__(self, x):
        return (x < self.thr).to(x.dtype)

    def __repr__(self):
        return f"ThresholdTransform({self.thr})"


def reshape_to_square(image, desired_size, color=(255, 255, 255), verbose=False):
    old_image_height, old_image_width, channels = image.shape
    ratio = old_image_height / old_image_width
    if ratio < 1:
        old_image_width, old_image_height = int(desired_size * ratio), desired_size
    else:
        old_image_width, old_image_height = desired_size, int(desired_size / ratio)
    image = cv2.resize(image, (old_image_height, old_image_width))

    # create new image of desired size and color (blue) for padding
    result = np.full((desired_size, desired_size, channels), color, dtype=np.uint8)
    # compute center offset
    x_center = (desired_size - old_image_width) // 2
    y_center = (desired_size - old_image_height) // 2
    # copy img image into center of result image
    result[x_center : x_center + old_image_width, y_center : y_center + old_image_height] = image
    if verbose:
        return result, ratio
    else:
        return result


def padding(img, size):
    h, w, c = img.shape
    img_pad = np.pad(
        img,
        (
            (size - (h % size) if h % size != 0 else 0, 0),
            (0, (size - (w % size) if w % size != 0 else 0)),
            (0, 0),
        ),
        mode="constant",
        constant_values=255,
    )
    return img_pad


def draw_bounding_boxes(img, box, width=2):
    img = img.copy()
    box = box.copy()
    from torchvision.utils import draw_bounding_boxes

    img_width = img.shape[0]
    box[:, [1, 3]] -= 1
    box[:, [1, 3]] *= -1
    box[:, [1, 3]] = box[:, [3, 1]]
    box = torch.tensor(box)
    transform = transforms.Compose([transforms.ToImage(), transforms.ToDtype(torch.uint8)])
    return draw_bounding_boxes(transform(img), boxes=box * img_width, colors="red", width=width)


def draw_point(img, box, width=4, color=(0, 255, 0)):
    alpha = None
    if img.shape[2] == 4:
        alpha = img[:, :, 3:]
    if isinstance(img, np.ndarray):
        img = img[:, :, :3].copy()
    elif isinstance(img, torch.Tensor):
        img = img.cpu().detach().numpy()
        transform = transforms.Compose([transforms.ToImage(), transforms.ToDtype(torch.uint8)])
        img = transform(img)
    if isinstance(box, np.ndarray):
        box = box.copy()
    elif isinstance(box, torch.Tensor):
        box = box.detach().cpu().numpy()
    else:
        box = np.array(box)
    img_width = img.shape[1]
    img_height = img.shape[0]
    box[..., 1] = 1 - box[..., 1]
    box[..., 0] *= img_width
    box[..., 1] *= img_height
    box = box.astype(np.int32)
    for b in box:
        cv2.circle(img, (b[0], b[1]), width, color, -1)
    if alpha is not None:
        img = np.concatenate([img, alpha], axis=2)
    return img


def draw_rect(img, box, width=4, color=(0, 255, 0), scale=True):
    alpha = None
    if img.shape[2] == 4:
        alpha = img[:, :, 3:]
    if isinstance(img, np.ndarray):
        img = img[:, :, :3].copy()
    elif isinstance(img, torch.Tensor):
        img = img.cpu().detach().numpy()
        transform = transforms.Compose([transforms.ToImage(), transforms.ToDtype(torch.uint8)])
        img = transform(img)
    if isinstance(box, np.ndarray):
        box = box.copy()
    elif isinstance(box, torch.Tensor):
        box = box.detach().cpu().numpy()
    else:
        box = np.array(box)
    img_height, img_width = img.shape[0], img.shape[1]
    if scale:
        box[..., 1] = 1 - box[..., 1]
        box[..., 0] *= img_width
        box[..., 1] *= img_height
    else:
        box[..., 1] = img.shape[0] - box[..., 1]
    box = box.astype(np.int32)
    for b in box:
        # cv2.circle(img, (b[0], b[1]), width, color, -1)
        start_point = (max(b[0] - width / 2, 0), max(b[1] - width / 2, 0))
        end_point = (b[0] + width / 2, b[1] + width / 2)
        start_point = tuple(map(int, start_point))
        end_point = tuple(map(int, end_point))
        cv2.rectangle(img, start_point, end_point, color, -1)
    if alpha is not None:
        img = np.concatenate([img, alpha], axis=2)
    return img


def draw_lines(img, lines):
    if isinstance(img, np.ndarray):
        img = img.copy()
    for line in lines:
        img = draw_line(img, line)
    return img


def draw_line(
    img, box, color=(0, 0, 255), thickness=2, endpoint=False, endpoint_thickness=2, scale=True
):
    if len(box) == 0:
        return img
    alpha = None
    if img.shape[2] == 4:
        alpha = img[:, :, 3:]
    if isinstance(img, np.ndarray):
        img = img[:, :, :3].copy()
    elif isinstance(img, torch.Tensor):
        img = img.cpu().detach().numpy()
        transform = transforms.Compose([transforms.ToImage(), transforms.ToDtype(torch.uint8)])
        img = transform(img)
    if isinstance(box, np.ndarray):
        box = box.copy()
    elif isinstance(box, torch.Tensor):
        box = box.detach().cpu().numpy()
    else:
        box = np.array(box)
    img_width = img.shape[1]
    img_height = img.shape[0]
    if scale:
        box[..., 1] = 1 - box[..., 1]
        box[..., 0] *= img_width
        box[..., 1] *= img_height
    else:
        box[..., 1] = img.shape[0] - box[..., 1]
    box = box.astype(np.int32)
    for b in box:
        cv2.line(img, b[0], b[1], color, thickness)
        if endpoint:
            cv2.circle(img, b[0], endpoint_thickness, color, -1)
            cv2.circle(img, b[1], endpoint_thickness, color, -1)
    if alpha is not None:
        img = np.concatenate([img, alpha], axis=2)
    return img


def create_grid(images, **args):
    p = []
    for image in images:
        s = torch.tensor(image)
        s = s.permute(2, 0, 1)
        p.append(s)
    p = torch.stack(p)
    fimg = make_grid(p, **args)
    fimg = fimg.permute(1, 2, 0).numpy()
    return fimg


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def get_attr(obj):
    return {
        x: getattr(obj, x)
        for x in dir(obj)
        if not (x.startswith("_") or inspect.ismodule(getattr(obj, x)))
    }


def Hungarian_Order(g1b, g2b, criterion):
    # cost matrix
    T = np.zeros((len(g1b[0]), len(g1b[0])))
    for idx, (g1, g2) in enumerate(zip(torch.as_tensor(g1b), torch.as_tensor(g2b))):
        for i, ix in enumerate(g1):
            for j, jx in enumerate(g2):
                T[i][j] = criterion(ix, jx)
        row_ind, col_ind = linear_sum_assignment(T)
        g2b[idx] = g2b[idx][row_ind][col_ind]


def take(sequence, axis):
    if axis == 0:
        yield from sequence
    else:
        for item in sequence:
            yield from take(item, axis - 1) if axis >= 0 else item


# def stratified_sampling(dataset: Dataset, train_samples_per_class: int):
#     import collections

#     train_indices = []
#     val_indices = []
#     target_counter = collections.Counter()
#     for idx, (data, target) in enumerate(dataset):
#         target_counter[target] += 1
#         if target_counter[target] <= train_samples_per_class:
#             train_indices.append(idx)
#         else:
#             val_indices.append(idx)
#     train_dataset = Subset(dataset, train_indices)
#     train_dataset = TensorDataset(
#         torch.stack([x[0] for x in train_dataset]),
#         torch.cat([torch.tensor([x[1]]) for x in train_dataset]),
#     )
#     val_dataset = Subset(dataset, val_indices)
#     val_dataset = TensorDataset(
#         torch.stack([x[0] for x in val_dataset]),
#         torch.cat([torch.tensor([x[1]]) for x in val_dataset]),
#     )
#     return train_dataset, val_dataset
