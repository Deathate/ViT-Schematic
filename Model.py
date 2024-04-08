import collections.abc as abc
import gc
import hashlib
import inspect
import io
import itertools
import json
import logging
import math
import os
import pickle
import random
import secrets
import shutil
import textwrap
import time
from functools import partial
from itertools import count
from pathlib import Path
from pprint import pprint
from typing import Annotated

import cv2
import cv2 as cv
import ipyplot
import IPython
import matplotlib.pyplot as plt
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
from PIL import Image
from plotly.subplots import make_subplots
from shapely import union
from shapely.geometry import LineString, MultiLineString, MultiPoint, Point
from sklearn.metrics import accuracy_score, classification_report
from tabulate import tabulate
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import Accuracy
from torchmetrics.classification import MulticlassAccuracy
from tqdm.std import tqdm

from utility import *
from visualizer import get_local


def select_gpu_with_most_free_memory():
    print()
    print("#" * 50)
    import pynvml

    pynvml.nvmlInit()
    deviceCount = pynvml.nvmlDeviceGetCount()
    print(f"GPU Available: {torch.cuda.is_available()}")
    print(f"CUDA_VISIBLE_DEVICES: {deviceCount}")
    memory = 0
    device = 0
    for i in range(deviceCount):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        print("- DEVICE:", i)
        print(f"  TOTAL: {int(info.total / 1024**2)}, FREE: {int(info.free / 1024**2)}")
        if info.free > memory:
            memory = info.free
            device = i
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
    print(f"Using GPU: [{device}]")
    print("#" * 50)
    print()


rng = np.random.default_rng()


class Datasetbehaviour:
    def __init__(self, size, creater: abc.Callable, *args):
        self.size = size
        self.creater = creater
        self.args = args
        key = inspect.getsource(type(self)) + str(args) + str(size)
        filepath = self.__get_filepath(key)
        self.filepath = str(filepath)
        self.__dataset = None
        self.RESET = False
        self.MP = False

    def __get_filepath(self, key: str):
        class_name = type(self).__name__
        parent = Path("custom_datasets") / Path(class_name)
        parent.mkdir(parents=True, exist_ok=True)
        id = hashlib.sha256(key.encode("utf-8")).hexdigest()
        file = Path(class_name + "_" + id + ".pkl")
        return parent / file

    def __load(self):
        print("[dataset creation]")
        if Path(self.filepath).exists() and not self.RESET:
            print("    [cache found]:")
            print(textwrap.fill(self.filepath, 70, initial_indent=" " * 4))
            self.__dataset = pickle.load(open(self.filepath, "rb"))
        else:
            if not self.MP:
                dataset = []
                for _ in tqdm(range(self.size)):
                    rng = np.random.default_rng()
                    dataset.append(self.creater(*self.args))
            else:
                def create():
                    rng = np.random.default_rng()
                    return self.creater(*self.args)

                dataset = p_tqdm.p_umap(
                    create,
                    range(self.size),
                    num_cpus=os.cpu_count() - 4,
                )
            self.__dataset = list(map(list, zip(*dataset)))
            pickle.dump(self.__dataset, open(self.filepath, "wb"))
        print("--- [finish creation] ---\n")
        self.loaded = True

    def __getitem__(self, idx):
        if self.__dataset is None:
            self.__load()
        if isinstance(idx, slice):
            return list(zip(self.__dataset[0][idx], self.__dataset[1][idx]))
        else:
            return self.__dataset[0][idx], self.__dataset[1][idx]

    def __len__(self):
        if self.__dataset is None:
            self.__load()
        return len(self.__dataset[0])

    def save_params(self):
        frame = inspect.currentframe()
        args, _, _, values = inspect.getargvalues(frame)
        arg_values = {arg: values[arg] for arg in set(args) - set(["self"])}
        for arg in arg_values:
            setattr(self, arg, arg_values[arg])

    def to_tensor(self, dataset, shape=(1, 1)):
        for i in range(len(dataset[0])):
            for j in range(len(shape)):
                if shape[j] > 1:
                    for k in range(shape[j]):
                        dataset[j][i][k] = torch.tensor(np.asarray(dataset[j][i][k]))
                else:
                    dataset[j][i] = torch.tensor(np.asarray(dataset[j][i]))

    def union_dataset(self, instance):
        if self.__dataset is None:
            self.__load()
        self.__dataset[0] += instance.dataset[0]
        self.__dataset[1] += instance.dataset[1]
        return self


class Model:
    def __init__(
        self,
        name,
        data,
        transform=None,
        ytransform=None,
        eval_metrics=None,
        batch_size=64,
        validation_split=0.1,
        shuffle=False,
    ):
        self.name = name
        self.transform = transform
        self.ytransform = ytransform
        if self.transform is None:
            self.transform = lambda x: torch.tensor(x).float()
        if self.ytransform is None:
            self.ytransform = lambda x: torch.tensor(x).float()
        select_gpu_with_most_free_memory()
        # torch.set_default_device('cuda')
        dataset = self.preprocessing(data)
        train_dataset, test_dataset = torch.utils.data.random_split(
            dataset,
            [tn := int(len(dataset) * (1 - validation_split)), len(dataset) - tn],
        )
        self.train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            # pin_memory=True,
            # num_workers=os.cpu_count(),
            # persistent_workers=True,
        )
        self.test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            # pin_memory=True,
            # num_workers=os.cpu_count(),
            # persistent_workers=True,
        )
        self.ep = 0
        self.total_time = 0
        self.eval_metrics = eval_metrics
        self.model = None
        self.model_id = None
        torch.cuda.empty_cache()
        self.interrupt = False
        self.tensorboard_setting()

    def tensorboard_setting(self):
        self.writer = SummaryWriter(comment=self.name)
        layout = {
            "metrics": {
                "loss": ["Multiline", ["Loss/train"]],
                "accuracy": ["Multiline", ["Loss/validation"]],
            },
        }
        self.writer.add_custom_scalars(layout)

    def preprocessing(self, data: Datasetbehaviour):
        print("[data preprocessing]")
        transform_id = data.filepath
        try:
            transform_id += inspect.getsource(self.transform)
        except:
            transform_id += str(self.transform)
        try:
            transform_id += inspect.getsource(self.ytransform)
        except:
            transform_id += str(self.ytransform)

        filepath = (
            Path(data.filepath).parent
            / "cache"
            / Path(hashlib.sha256(transform_id.encode("utf-8")).hexdigest())
        )
        if filepath.exists():
            print("    [cache found]:")
            print(textwrap.fill(str(filepath), 70, initial_indent=" " * 4))
            result = pickle.load(open(filepath, "rb"))
        else:
            filepath.parent.mkdir(parents=True, exist_ok=True)
            result = [[self.transform(x[0]), self.ytransform(x[1])] for x in tqdm(data)]
            pickle.dump(result, open(filepath, "wb"))
        result = [[x[0].cuda(non_blocking=True), x[1].cuda(non_blocking=True)] for x in result]
        print("--- [finish preprocessing] ---\n")
        return result

    def fit(self, model, criterion, optimizer, epochs=1, compile=False, amp=True, target_transform=lambda y_hat, y: y):
        if id(model) != self.model_id:
            self.model_id = id(model)
            self.gc()
            torch.backends.cudnn.benchmark = True
            self.model = model.to(device="cuda")
            self.model_overview(self.model)
            print(f"Model: {self.model.__class__.__name__}, ID:{self.model_id}")
            # accelerate trining speed
            if compile:
                self.model = torch.compile(self.model, mode="reduce-overhead")
            torch.set_float32_matmul_precision("high")
        if self.interrupt:
            return
        try:
            print("----------- Training started -----------")
            start_time = time.time()
            start = self.ep
            end = self.ep + epochs

            early_stopping_monitor = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15)
            if amp:
                scaler = torch.cuda.amp.GradScaler()
            for ep in range(start, end):
                self.model.train()
                with tqdm(
                    total=len(self.train_loader),
                    bar_format="{desc}{n_fmt}/{total_fmt}|{bar}| - {elapsed}s{postfix}",
                ) as pbar:
                    pbar.set_description(f"Epoch {ep+1}/{end}")
                    train_loss = []
                    pbar.set_postfix({"loss": "---", "acc": "---"})
                    for data, target in self.train_loader:
                        def train():
                            y_hat = self.model(data, target)
                            if ep == 0:
                                from torchviz import make_dot
                                graph = make_dot(y_hat, params=dict(model.named_parameters()))
                                graph.render(Path(self.writer.log_dir) /
                                             "model_graph", format="png")
                                # png_graph = graph.pipe(format='png')
                                # png_graph = Image.open(io.BytesIO(png_graph))
                                # png_graph = np.array(png_graph)
                                # self.writer.add_image("Model Graph", png_graph, dataformats='HWC')
                            try:
                                loss = self.loss(y_hat, target, criterion, eval=False,
                                                 target_transform=target_transform)
                                optimizer.zero_grad()
                                if amp:
                                    scaler.scale(loss).backward()
                                else:
                                    loss.backward()
                                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                                optimizer.step()
                                train_loss.append(loss)
                                pbar.update(1)
                            except Exception as e:
                                # print("Error in loss calculation", e)
                                print("!!!-------- Error captured --------!!!")
                                error = tabulate(
                                    [
                                        ["y_hat", y_hat.dtype, list(y_hat.shape)],
                                        ["target", target.dtype, list(target.shape)],
                                    ],
                                    headers=["", "dtype", "shape"],
                                    tablefmt="psql",
                                )
                                print(error)
                                print(e)
                                exit()
                        if amp:
                            with torch.cuda.amp.autocast():
                                train()
                        else:
                            train()

                    train_loss = torch.stack(train_loss).mean()
                    self.writer.add_scalar("Loss/train", train_loss, ep + 1)
                    # calculate validation loss
                    self.model.eval()
                    val_loss = []
                    for data, target in self.test_loader:
                        y_hat = self.predict(data, target)
                        val_loss.append(self.loss(y_hat, target, criterion,
                                        eval=True, target_transform=target_transform))
                    val_loss = torch.stack(val_loss).mean()
                    self.writer.add_scalar("Loss/validation", val_loss, ep + 1)
                    # self.writer.add_scalars('Loss', {'validation': val_loss,
                    #                                  'train': train_loss}, ep + 1)
                    pbar.set_postfix({"loss": f"{train_loss:.3E}", "acc": f"{val_loss:.3E}"})
                    early_stopping_monitor.step(train_loss)
                    if early_stopping_monitor.num_bad_epochs >= early_stopping_monitor.patience:
                        print("Early Stopping")
                        break
            self.ep = ep + 1
        except KeyboardInterrupt:
            print("Keyboard interrupt received.")
            self.interrupt = True

        end_time = time.time()
        print(f"Elapsed time: {end_time - start_time + self.total_time:.3f} seconds")
        self.total_time += end_time - start_time
        print("----------- Training finished -----------")
        print()
        torch.save(model.state_dict(), Path(self.writer.log_dir) / "checkpoint.pth")
        shutil.copy(inspect.getfile(self.model.__class__), Path(self.writer.log_dir))

    @torch.no_grad()
    def predict(self, data: torch.tensor, target: torch.tensor):
        return self.model(data, target)

    def __call__(self, data, target):
        return self.predict(data.unsqueeze(0).unsqueeze(0).cuda(), target.unsqueeze(0).unsqueeze(0).cuda())

    def inference(self, testset):
        self.model.eval()
        testset = self.preprocessing(testset)
        loader = DataLoader(dataset=testset, batch_size=len(testset))
        x = next(iter(loader))[0]
        y = next(iter(loader))[1]
        return list(zip(x, self.predict(x, y).cpu(), y.cpu()))

    def loss(self, y_hat, y, criterion, eval, target_transform):
        y = target_transform(y_hat, y)
        if eval and self.eval_metrics:
            return self.eval_metrics(criterion, y_hat, y)
        else:
            return criterion(y_hat, y)

    @property
    def weight(self):
        return "\n".join(map(lambda x: str(x), self.model.named_parameters()))

    def model_overview(self, model):
        from torchinfo import summary

        s = summary(model, row_settings=("var_names",))
        c = count()
        table = []
        params_num = 0
        for x in s.summary_list:
            if x.depth == 1:
                table.append([next(c), x.var_name, type(x.module).__name__, x.num_params])
                params_num += x.num_params
        if s.trainable_params - params_num > 0:
            table.append([next(c), "other", "---", s.trainable_params - params_num])
        table.append(["Total", "", "", s.total_params])
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        size_all_mb = (param_size + buffer_size) / 1024**2
        table.append(["Size (MB)", "", "", f"{size_all_mb:.3f}"])
        print(
            tabulate(
                table,
                headers=["", "Name", "Type", "Params"],
                tablefmt="psql",
                numalign="right",
            )
        )

    def save(self, name):
        torch.save(self.model.state_dict(), name)

    def load(self, name):
        self.model.load_state_dict(torch.load(name))

    def __getitem__(self, size):
        loader = next(iter(self.train_loader))
        ret = list(zip(loader[0][size].cpu(), loader[1][size].cpu()))
        if isinstance(size, slice):
            return ret
        else:
            return ret[0]

    def gc(self):
        collected = gc.collect()
        print(f"Garbage collector: collected {collected} objects.")
        torch.cuda.empty_cache()
        self.ep = 0
        self.interrupt = False
        self.tensorboard_setting()
        self.total_time = 0


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


# @torch.no_grad()
# def Hungarian_Order(g1b, g2b):
#     from hungarian_algorithm import algorithm
#     result = []
#     g1b = torch.as_tensor(g1b)
#     g2b = torch.as_tensor(g2b)
#     for idx, (g1, g2) in enumerate(zip(g1b.cpu(), g2b.cpu())):
#         print(g1)
#         print(g2)
#         G = {}
#         for i, ix in enumerate(g1):
#             G[f"{i}"] = {j: (torch.square(
#                 ix-jx).sum() + 1).item() for j, jx in enumerate(g2)}

#         order = algorithm.find_matching(G, matching_type='min',
#                                         return_type='list')
#         order = sorted([x[0] for x in order])
#         new_match = g2b[idx][[x[1] for x in order]]
#         result.append(new_match)


#     return torch.stack(result)
def Hungarian_Order(g1b, g2b):
    from scipy.optimize import linear_sum_assignment

    # cost matrix
    T = np.zeros((len(g1b), len(g1b[0]), len(g1b[0])))
    for idx, (g1, g2) in enumerate(zip(torch.as_tensor(g1b), torch.as_tensor(g2b))):
        for i, ix in enumerate(g1):
            for j, jx in enumerate(g2):
                T[idx][i][j] = torch.square(ix - jx).sum()
        row_ind, col_ind = linear_sum_assignment(T[idx])
        g2b[idx] = g2b[idx][col_ind]

    return g2b


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        """
        Inputs
            d_model - Hidden dimensionality of the input.
            max_len - Maximum length of a sequence to expect.
        """
        super().__init__()
        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x):
        return self.pe[:, : x.size(1)]


def plotly_to_array(fig):
    image_bytes = fig.to_image(format="jpg")
    return np.asarray(Image.open(io.BytesIO(image_bytes)))


def seaborn_to_array(fig):
    return np.asarray(fig.get_figure().canvas.buffer_rgba())


def visualize_attentions(maps):
    map_num = len(maps[0]) - 1
    fig = make_subplots(rows=1, cols=map_num + 1, horizontal_spacing=0)

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


def flatten_list(lst):
    flattened_list = []
    for item in lst:
        if isinstance(item, list) or isinstance(item, tuple):
            flattened_list.extend(flatten_list(item))
        else:
            flattened_list.append(item)
    return flattened_list


def plot_images(images, img_width=None, max_images=5):
    if not isinstance(images, abc.Sequence) and not isinstance(images, Datasetbehaviour) and not isinstance(images, Model):
        images = [images]
    images = images[:max_images]
    L = len(images)
    images = flatten_list(images)

    for i in range(len(images)):
        if isinstance(images[i], torch.Tensor):
            scale = images[i].max() > 1 and images[i].dtype == torch.float32
            images[i] = np.array(transforms.ToPILImage()(images[i]))
            if scale:
                images[i] = 255 - images[i]
            # if images[i].max() <= 1:
            #     images[i] = images[i] * 255
        elif images[i].max() <= 1:
            # images[i] = images[i] * 255
            images[i] = images[i].astype(np.uint8)

    cols = len(images) // L
    for i in range(0, len(images), cols):
        ipyplot.plot_images(
            images[i: i + cols],
            img_width=images[i].shape[0] if img_width == -
            1 else img_width if img_width else 200,
        )


class ThresholdTransform(object):
    def __init__(self, thr_255):
        self.thr = thr_255

    def __call__(self, x):
        return (x < self.thr).to(x.dtype)

    def __repr__(self):
        return f"ThresholdTransform({self.thr})"


def reshape_to_square(image, desired_size, color=(255, 255, 255)):
    old_image_height, old_image_width, channels = image.shape
    ratio = old_image_height / old_image_width
    if ratio < 1:
        old_image_width, old_image_height = int(desired_size * ratio), desired_size
    else:
        old_image_width, old_image_height = desired_size, int(desired_size / ratio)
    image = cv.resize(image, (old_image_height, old_image_width))

    # create new image of desired size and color (blue) for padding
    result = np.full((desired_size, desired_size, channels), color, dtype=np.uint8)
    # compute center offset
    x_center = (desired_size - old_image_width) // 2
    y_center = (desired_size - old_image_height) // 2
    # copy img image into center of result image
    result[x_center:x_center + old_image_width, y_center:y_center + old_image_height] = image

    return result
