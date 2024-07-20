import collections.abc as abc
import copy
import os
import datetime
import gc
import glob
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
import typing
from collections import OrderedDict, namedtuple
from dataclasses import dataclass
from functools import partial
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
from IPython.display import display
from PIL import Image, ImageOps
from plotly.subplots import make_subplots
from scipy.optimize import linear_sum_assignment
from shapely import ops, union, union_all
from shapely.geometry import LineString, MultiLineString, MultiPoint, Point, Polygon
from sklearn.metrics import accuracy_score, classification_report
from tabulate import tabulate
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import Accuracy
from torchmetrics.classification import MulticlassAccuracy
from torchvision import models
from torchvision.ops.boxes import complete_box_iou
from tqdm import tqdm

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
def set_seed(seed):
    # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    # torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

class DataCell(typing.NamedTuple):
    input: object
    output: object
    meta: dict = None


class Datasetbehaviour:
    def __init__(self, size, creater: abc.Callable, *args, **kwargs):
        self.size = size
        self.creater = lambda: creater(*args, **kwargs)
        self.reset = Datasetbehaviour.RESET
        key = inspect.getsource(type(self)) + str(args) + str(kwargs) + str(size)
        filepath = self.__get_filepath(key)
        print("** [Create dataset]")
        print("[Identify file path]")
        print(textwrap.fill(str(filepath), 70, initial_indent=" " * 4))
        self.filepath = str(filepath)
        self.__dataset = None

    def __get_filepath(self, key: str):
        class_name = type(self).__name__
        parent = Path("custom_datasets") / Path(class_name)
        parent.mkdir(parents=True, exist_ok=True)
        id = hashlib.sha256(key.encode("utf-8")).hexdigest()
        file = Path(class_name + "_" + id + ".pkl")
        return parent / file

    def __load(self):
        print("[Load dataset]")
        print("- New:", Datasetbehaviour.RESET)
        print("- Multiple Processes:", Datasetbehaviour.MP)
        if Path(self.filepath).exists() and not Datasetbehaviour.RESET:
            print("** [cache found]")
            print(textwrap.fill(self.filepath, 70, initial_indent=" " * 4))
            self.__dataset = pickle.load(open(self.filepath, "rb"))
        else:
            if not Datasetbehaviour.MP:
                dataset = []
                for _ in tqdm(range(self.size)):
                    dataset.append(self.creater())
            else:
                dataset = p_tqdm.p_umap(
                    lambda _: self.creater(),
                    range(self.size),
                    num_cpus=os.cpu_count() - 4,
                )
            self.__dataset = np.array(dataset, dtype=object)
            pickle.dump(self.__dataset, open(self.filepath, "wb"))
        print("--- [Loading done] ---\n")
        Datasetbehaviour.reset()

    def __getitem__(self, idx):
        if self.__dataset is None:
            self.__load()
        contain_slice = False
        if isinstance(idx, slice):
            contain_slice = True
        if isinstance(idx, tuple):
            for i in idx:
                if isinstance(i, slice):
                    contain_slice = True
                    break
        if contain_slice:
            return self.__dataset[idx].tolist()

        return self.__dataset[idx]

    def __len__(self):
        if self.__dataset is None:
            self.__load()
        return len(self.__dataset)

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

    def dataset(self):
        return self.__dataset

    def union_dataset(self, instance):
        if self.__dataset is None:
            self.__load()
        self.__dataset[0] += instance.dataset()[0]
        self.__dataset[1] += instance.dataset()[1]
        return self

    def reset():
        Datasetbehaviour.MP = False
        Datasetbehaviour.RESET = False

    def view(self):
        pprint(self[0])


Datasetbehaviour.reset()


def cudalization(x):
    if isinstance(x, torch.Tensor):
        return x.cuda(non_blocking=True)
    else:
        return [cudalization(y) for y in x]


class Model:
    def __init__(
        self,
        name,
        data,
        batch_size=64,
        xtransform=None,
        ytransform=None,
        validation_split=0.1,
        shuffle=True,
        amp=True,
        suffix="",
        cudalize=True,
        use_cache=True,
        memory_fraction=1,
        eval=False,
    ):
        if memory_fraction < 1:
            torch.cuda.set_per_process_memory_fraction(memory_fraction)
        self.name = name
        self.xtransform = xtransform
        self.ytransform = ytransform
        self.suffix = suffix
        if self.xtransform is None:
            self.xtransform = lambda x: torch.tensor(x).float().cuda()
        if self.ytransform is None:
            self.ytransform = lambda x: torch.tensor(x).float().cuda()
        if torch.cuda.device_count() > 1:
            select_gpu_with_most_free_memory()
        # torch.set_default_device('cuda')
        self.cudalize = cudalize
        self.data = data
        if not eval:
            self.dataset = self.preprocessing(data, use_cache, cudalize)
            self.train_dataset, self.test_dataset = torch.utils.data.random_split(
                self.dataset,
                [tn := int(len(self.dataset) * (1 - validation_split)), len(self.dataset) - tn],
            )
            self.train_loader = DataLoader(
                dataset=self.train_dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                # pin_memory=True,
                # num_workers=os.cpu_count(),
                # persistent_workers=True,
            )
            self.test_loader = DataLoader(
                dataset=self.test_dataset,
                batch_size=batch_size,
                # pin_memory=True,
                # num_workers=os.cpu_count(),
                # persistent_workers=True,
            )
        self.model = None
        self.model_id = None

        self.amp = amp
        self.writer = None
        torch.set_float32_matmul_precision("high")

    def tensorboard_setting(self):
        if self.writer:
            self.writer.close()
        self.writer = SummaryWriter(comment="", log_dir=self.log_dir)
        layout = {
            "metrics": {
                "loss": ["Multiline", ["Loss/train"]],
                "accuracy": ["Multiline", ["Loss/validation"]],
                "learning rate": ["Multiline", ["lr"]],
            },
        }
        self.writer.add_custom_scalars(layout)

    def preprocessing(self, data: Datasetbehaviour, use_cache, cudalize):
        print("[data preprocessing]")
        load_from_cache = use_cache and not data.reset
        if load_from_cache:
            transform_id = data.filepath
            try:
                transform_id += inspect.getsource(self.xtransform)
            except:
                transform_id += str(self.xtransform)
            try:
                transform_id += inspect.getsource(self.ytransform)
            except:
                transform_id += str(self.ytransform)

            filepath = (
                Path(data.filepath).parent
                / "cache"
                / Path(hashlib.sha256(transform_id.encode("utf-8")).hexdigest())
            )
        if load_from_cache and filepath.exists():
            print("** [cache found]")
            result = pickle.load(open(filepath, "rb"))
        else:
            try:
                result = [
                    [i, self.xtransform(x[0]), self.ytransform(x[1])]
                    for i, x in enumerate(tqdm(data))
                ]
            except Exception as e:
                print("Error in transformation")
                raise (e)

            if load_from_cache:
                filepath.parent.mkdir(parents=True, exist_ok=True)
                pickle.dump(result, open(filepath, "wb"))
        # result = [[x[0].cuda(non_blocking=True), x[1].cuda(non_blocking=True)] for x in result]
        if cudalize:
            print("*cudalized")
            for r in tqdm(result):
                r[1] = cudalization(r[1])
                r[2] = cudalization(r[2])
        print("--- [finish preprocessing] ---\n")
        return result

    def fit(
        self,
        model,
        criterion,
        optimizer,
        epochs=1,
        max_epochs=99999,
        start_epoch=None,
        compile=False,
        target_transform=lambda y_hat, y: y,
        early_stopping=False,
        eval_metrics=None,
        training_epoch_end=None,
        pretrained_path=None,
        keep=True,
        backprop_freq=1,
        device_ids=[0],
        keep_epoch=True,
    ):
        now = datetime.datetime.now()
        self.log_dir = Path(
            f"runs/{self.name}/" + now.strftime("%m%d_%H-%M-%S") + "_" + self.suffix + "/"
        )
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, device_ids))
        backprop_freq = int(backprop_freq)
        previous_epoch = 0
        list_of_files = glob.glob(str(self.log_dir.parent) + "/*")
        if pretrained_path == "latest":
            if len(list_of_files) == 0:
                pretrained_path = ""
            else:
                latest_file = max(list_of_files, key=os.path.getctime)
                pretrained_path = latest_file + "/latest.pth"
                if not Path(pretrained_path).exists():
                    pretrained_path = ""
        if pretrained_path != "":
            print(f'** [Pretrained model loaded] - "{pretrained_path}"')
            checkpoint = torch.load(pretrained_path)
            if isinstance(checkpoint, OrderedDict):
                model.load_state_dict(checkpoint, strict=False)
                model = self.parallel(model, device_ids)
            else:
                lr = optimizer.param_groups[0]["lr"]
                # it is recommended to move a model to GPU before constructing an optimizer
                model.load_state_dict(checkpoint["model"], strict=False)
                model = self.parallel(model, device_ids)
                print("** [Pretrained optimizer loaded]")
                optimizer.load_state_dict(checkpoint["optimizer"])
                if optimizer.param_groups[0]["lr"] != lr:
                    print(f"** [Optimizer learning rate changed to {lr}]")
                    optimizer.param_groups[0]["lr"] = lr
                if checkpoint.get("epoch", False) and keep_epoch:
                    previous_epoch = checkpoint["epoch"] + 1

        else:
            model = self.parallel(model, device_ids)
        self.model = model
        # accelerate training speed
        if compile:
            self.model = torch.compile(self.model, mode="reduce-overhead")
        if not keep:
            return
        if id(model) != self.model_id:
            self.model_id = id(model)
            self.gc()
            self.tensorboard_setting()
            shutil.copy(inspect.getfile(self.model.__class__), Path(self.writer.log_dir))
            self.model_overview(self.model)
            print(f"Model: {self.model.__class__.__name__}, ID:{self.model_id}")
        if self.interrupt:
            return
        try:
            best_quantity = 1e5
            print("----------- Training started -----------")
            start_time = time.time()
            if start_epoch is not None:
                self.ep = start_epoch
            start = self.ep
            if isinstance(max_epochs, int):
                max_epochs = int(max_epochs)
            end = min(max_epochs, start + previous_epoch + epochs)

            if self.amp:
                scaler = torch.cuda.amp.GradScaler()
            early_stopping_monitor = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=30)
            # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, min_lr=1e-6)

            self.model.train()
            Meta_data = namedtuple("Meta_data", ["data", "model"])

            def create_meta(seq):
                return Meta_data(
                    [DataCell(*self.data[s]) for s in seq],
                    self.model if len(device_ids) == 1 else self.model.module,
                )

            for ep in range(start + previous_epoch, min(end, start + previous_epoch + epochs)):

                def save_model(path, log=False):
                    torch.save(
                        {
                            "model": (
                                self.model.state_dict()
                                if len(device_ids) == 1
                                else self.model.module.state_dict()
                            ),
                            "optimizer": optimizer.state_dict(),
                            "epoch": ep,
                        },
                        path,
                    )
                    if log:
                        pbar.write(f'Saved to "{path}".')

                with tqdm(
                    total=len(self.train_loader) // backprop_freq,
                    bar_format="{desc}{n_fmt}/{total_fmt}|{bar}| - {elapsed}s{postfix}",
                ) as pbar:
                    pbar.set_description(f"Epoch {ep+1}/{end} ({max_epochs})")
                    train_loss = []
                    pbar.set_postfix({"loss": "---", "acc": "---"})

                    try:
                        self.model.train()
                        accum_loss = 0
                        for batch_num, (seq, data, target) in enumerate(self.train_loader):
                            if not self.cudalize:
                                data = cudalization(data)
                                target = cudalization(target)

                            if self.amp:
                                with torch.cuda.amp.autocast():
                                    y_hat = self.model(data, target)
                                    loss = (
                                        self.loss(
                                            y_hat,
                                            target,
                                            criterion,
                                            eval=False,
                                            target_transform=target_transform,
                                            eval_metrics=eval_metrics,
                                            meta=create_meta(seq),
                                        )
                                        / backprop_freq
                                    )
                                    scaler.scale(loss).backward()
                                    accum_loss += loss.item()
                                    # pbar.write(f"loss: {loss.item()}")
                            else:
                                y_hat = self.model(data, target)
                                loss = (
                                    self.loss(
                                        y_hat,
                                        target,
                                        criterion,
                                        eval=False,
                                        target_transform=target_transform,
                                        eval_metrics=eval_metrics,
                                        meta=create_meta(seq),
                                    )
                                    / backprop_freq
                                )
                                loss.backward()
                                accum_loss += loss.item()
                            if (batch_num + 1) % backprop_freq == 0:
                                if self.amp:
                                    scaler.step(optimizer)
                                    scaler.update()
                                    optimizer.zero_grad()
                                else:
                                    optimizer.step()
                                    optimizer.zero_grad()
                                # if ep == 0:
                                #     from torchviz import make_dot
                                #     graph = make_dot(y_hat, params=dict(model.named_parameters()))
                                #     graph.render(Path(self.writer.log_dir) /
                                #                  "model_graph", format="png")

                                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                                pbar.set_postfix({"loss": f"{accum_loss:.3E}", "acc": "---"})
                                pbar.update(1)
                                train_loss.append(accum_loss)
                                accum_loss = 0
                        self.model.eval()
                        if training_epoch_end:
                            training_epoch_end()
                    except Exception as e:
                        # print("Error in loss calculation", e)
                        print("!!!-------- Error captured --------!!!")
                        try:
                            error = tabulate(
                                [
                                    ["y_hat", y_hat.dtype, list(y_hat.shape)],
                                    ["target", target.dtype, list(target.shape)],
                                ],
                                headers=["", "dtype", "shape"],
                                tablefmt="psql",
                            )
                            print(error)
                        except:
                            pass
                        print("Removing log directory")
                        shutil.rmtree(self.log_dir)
                        raise e

                    train_loss = np.mean(train_loss)
                    self.writer.add_scalar("Loss/train", train_loss, ep + 1)
                    # calculate validation loss
                    val_loss = []
                    for seq, data, target in self.test_loader:
                        if not self.cudalize:
                            data = cudalization(data)
                            target = cudalization(target)
                        y_hat = self.predict(data, target)
                        val_loss.append(
                            self.loss(
                                y_hat,
                                target,
                                criterion,
                                eval=True,
                                target_transform=target_transform,
                                eval_metrics=eval_metrics,
                                meta=create_meta(seq),
                            )
                        )
                    val_loss = torch.stack(val_loss).mean()
                    self.writer.add_scalar("Loss/validation", val_loss, ep + 1)
                    latest_path = Path(self.log_dir) / "latest.pth"
                    save_model(latest_path)
                    # self.writer.add_scalars('Loss', {'validation': val_loss,
                    #                                  'train': train_loss}, ep + 1)
                    pbar.set_postfix({"loss": f"{train_loss:.3E}", "acc": f"{val_loss:.3E}"})
                    if val_loss <= best_quantity:
                        best_quantity = val_loss
                        saved_path = Path(self.log_dir) / "best.pth"
                        save_model(saved_path)
                        pbar.write(
                            f'Found a better solution at epoch {ep+1}. Saved to "{saved_path}".'
                        )
                    # scheduler.step(val_loss)
                    self.writer.add_scalar("lr", optimizer.param_groups[0]["lr"], ep + 1)
                    if early_stopping:
                        early_stopping_monitor.step(train_loss)
                        if early_stopping_monitor.num_bad_epochs >= early_stopping_monitor.patience:
                            print("Early Stopping")
                            break

            self.ep = ep + 1
        except KeyboardInterrupt:
            print("Keyboard interrupt received.")
            self.interrupt = True
            if ep < 5:
                print("Removing log directory")
                shutil.rmtree(self.writer.log_dir)

        end_time = time.time()
        print(f"Elapsed time: {end_time - start_time + self.total_time:.3f} seconds")
        self.total_time += end_time - start_time
        print("----------- Training finished -----------")
        print()

    def parallel(self, model, device_ids):
        if len(device_ids) > 1:
            return nn.DataParallel(model, device_ids=list(range(len(device_ids)))).cuda()
        else:
            return model.cuda()

    @torch.no_grad()
    def predict(self, data: torch.tensor, target: torch.tensor):
        return self.model(data, target)

    # def __call__(self, data, target):
    #     return self.predict(self.xtransform(data), self.ytransform(target).unsqueeze(0).unsqueeze(0).cuda())

    @torch.no_grad()
    def inference(self, testset, preprocessing=False, verbose=True):
        print("[inference]")
        self.model.eval()
        if not preprocessing:
            testset = self.preprocessing(testset, True, True)
        loader = DataLoader(dataset=testset, batch_size=len(testset))
        x = next(iter(loader))[1]
        y = next(iter(loader))[2]
        if verbose:
            ret = list(zip(x, self.predict(x, y), y))
        else:
            ret = self.predict(x, y)
        return ret

    # def __call__(self, x, y) -> gc.Any:
    #     x = self.xtransform(x)
    #     y = self.ytransform(y)
    #     return self.model(x, y)

    def loss(self, y_hat, y, criterion, eval, target_transform, eval_metrics, meta):
        try:
            y = target_transform(y_hat, y)
        except Exception as e:
            if self.amp and isinstance(e, ValueError):
                pass
            else:
                raise (e)
            # pass
        if eval and eval_metrics:
            return eval_metrics(criterion, y_hat, y, meta)
        else:
            return criterion(y_hat, y, meta)

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
        return self.dataset[size]

    def view(self):
        pprint(self[0])

    def gc(self):
        collected = gc.collect()
        print(f"Garbage collector: collected {collected} objects.")
        torch.cuda.empty_cache()
        self.ep = 0
        self.interrupt = False
        self.total_time = 0


def loss_func(criterion):
    return lambda y_hat, y, meta: criterion(y_hat, y)


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
def Hungarian_Order(g1b, g2b, criterion):
    # cost matrix
    T = np.zeros((len(g1b[0]), len(g1b[0])))
    for idx, (g1, g2) in enumerate(zip(torch.as_tensor(g1b), torch.as_tensor(g2b))):
        for i, ix in enumerate(g1):
            for j, jx in enumerate(g2):
                T[i][j] = criterion(ix, jx)
        row_ind, col_ind = linear_sum_assignment(T)
        g2b[idx] = g2b[idx][row_ind][col_ind]


#     return g2b
# def Hungarian_Order(g1b, g2b, criterion):
#     # cost matrix
#     T = np.zeros((len(g1b[0]), len(g1b[0])))
#     indices = []
#     for idx, (g1, g2) in enumerate(zip(torch.as_tensor(g1b), torch.as_tensor(g2b))):
#         for i, ix in enumerate(g1):
#             for j, jx in enumerate(g2):
#                 T[i][j] = criterion(ix, jx)
#         indices.append(linear_sum_assignment(T))
#     return [
#         (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
#         for i, j in indices
#     ]


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


def plot_images(images, img_width=None, max_images=5):
    if (
        not isinstance(images, abc.Sequence)
        and not isinstance(images, Datasetbehaviour)
        and not isinstance(images, Model)
        and (isinstance(images, np.ndarray) and len(images.shape) == 3)
    ):
        images = [images]
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
        for i in range(0, len(images), cols):
            height = images[i].shape[0] if img_width == -1 else img_width if img_width else 200
            ipyplot.plot_images(
                images[i : i + cols],
                img_width=height,
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
    return draw_bounding_boxes(transform(img), boxes=box * img_width, colors="red", width=  width)


def draw_point(img, box, width=4):
    if isinstance(img, np.ndarray):
        img = img.copy()
    elif isinstance(img, torch.Tensor):
        img = img.cpu().detach().numpy()
        transform = transforms.Compose([transforms.ToImage(), transforms.ToDtype(torch.uint8)])
        img = transform(img)
    if isinstance(box, np.ndarray):
        box = box.copy()
    else:
        box = box.detach().cpu().numpy()
    img_width = img.shape[1]
    img_height = img.shape[0]
    box[..., 1] = 1 - box[..., 1]
    box[..., 0] *= img_width
    box[..., 1] *= img_height
    box = box.astype(np.int32)
    for b in box:
        cv2.circle(img, (b[0], b[1]), width, (0, 255, 0), -1)
    return img


def draw_lines(img, lines):
    if isinstance(img, np.ndarray):
        img = img.copy()
    for line in lines:
        img = draw_line(img, line)
    return img


def draw_line(img, box):
    if isinstance(img, np.ndarray):
        img = img.copy()
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
        cv2.line(img, b[0], b[1], (0, 0, 255), 2)
    return img


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]
