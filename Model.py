
import math
from torchmetrics.classification import MulticlassAccuracy
from tabulate import tabulate
import inspect
import IPython
import einops
import secrets
import p_tqdm
import collections.abc as abc
import matplotlib.pyplot as plt
from shapely import union
from shapely.geometry import MultiLineString, LineString, Point, MultiPoint
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import ipyplot
from typing import Annotated
import io
from PIL import Image
import shapely
import cv2 as cv
from utility import *
from functools import partial
from itertools import count
import pickle
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import tqdm
from pathlib import Path
import os
from torchmetrics import Accuracy
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset, Subset, Dataset
import torch.nn.functional as F
import torchvision.transforms.v2 as transforms
import torchvision
import torch.optim as optim
import torch.nn as nn
import torch
import logging
import hashlib
import seaborn

from visualizer import get_local
# get_local.activate()


def select_gpu_with_most_free_memory():
    print("#"*50)
    import pynvml
    pynvml.nvmlInit()
    deviceCount = pynvml.nvmlDeviceGetCount()
    print(f"GPU Available: {torch.cuda.is_available()}")
    print("CUDA_VISIBLE_DEVICES: ", deviceCount)
    memory = 0
    device = 0
    for i in range(deviceCount):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        print("- DEVICE:", i)
        print(f"  TOTAL:", int(info.total/1024**2),
              ", FREE:", int(info.free/1024**2))
        if info.free > memory:
            memory = info.free
            device = i
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
    print(f"Using GPU: [{device}]")
    print("#"*50)
    print()
    torch.set_default_device('cuda')


select_gpu_with_most_free_memory()
rng = np.random.default_rng()


class Model:
    def __init__(self, name, model, data, transform=None, ytransform = None, target_transform=None, eval_metrics=None):
        self.name = name
        self.model = model.to(device='cuda')
        # accelerate trining speed
        # self.model = torch.compile(self.model)
        torch.set_float32_matmul_precision('high')
        self.transform = transform
        self.ytransform = ytransform
        if self.transform is None:
            self.transform = lambda x: torch.tensor(x).float()
        if self.ytransform is None:
            self.ytransform = lambda x: torch.tensor(x).float()
        self.dataset = self.preprocessing(
            data)
        self.target_transform = target_transform
        self.writer = SummaryWriter(comment=self.name)
        layout = {
            "metrics": {
                "loss": ["Multiline", ["loss/train"]],
                "accuracy": ["Multiline", ["loss/validation"]],
            },
        }
        self.writer.add_custom_scalars(layout)
        self.ep = 0
        self.eval_metrics = eval_metrics
        self.model_overview(self.model)

    def preprocessing(self, data: list):
        dataset = []
        for i in range(len(data)):
            x = data[i][0]
            y = data[i][1]
            dataset.append((self.transform(x), self.ytransform(y)))

        return dataset

    def fit(self, criterion, optimizer, epochs=1, batch_size=64, validation_split=0.1, validation_freq=1, shuffle=False):
        generator = torch.Generator(device='cuda')
        train_dataset, test_dataset = torch.utils.data.random_split(self.dataset, [tn := int(
            len(self.dataset)*(1-validation_split)), len(self.dataset)-tn], generator=generator)

        train_loader = DataLoader(dataset=train_dataset,
                                       batch_size=batch_size, shuffle=shuffle, generator=generator)
        test_loader = DataLoader(dataset=test_dataset,
                                      batch_size=batch_size, generator=generator)
        start = self.ep
        end = self.ep + epochs

        early_stopping_monitor = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20)
        for ep in range(start, end):
            self.model.train()
            with tqdm.tqdm(total=len(train_loader), bar_format='{desc}{n_fmt}/{total_fmt}|{bar}| - {elapsed}s{postfix}') as pbar:
                pbar.set_description(f"Epoch {ep+1}/{end}")
                train_loss = []
                for (data, target) in train_loader:
                    y_hat = self.model(data)
                    try:
                        loss = self.loss(
                            y_hat, target, criterion, eval=False)
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        optimizer.step()
                        train_loss.append(loss)
                        pbar.set_postfix(
                            {"loss": f"{loss:.3f}", "acc": f"{0:.3f}"})
                        pbar.update(1)
                    except Exception as e:
                        # print("Error in loss calculation", e)
                        error = tabulate([["y_hat", y_hat.dtype, list(y_hat.shape)], ["target", target.dtype, list(target.shape)]], headers=[
                                         "", "dtype", "shape"], tablefmt="psql")
                        print(error)
                        print(e)
                        exit()

                train_loss = torch.stack(train_loss).mean()
                self.writer.add_scalar(
                    "loss/train", train_loss, ep+1)

                # calculate validation loss
                self.model.eval()
                val_loss = []
                for (data, target) in test_loader:
                    y_hat = self.predict(
                        data)
                    val_loss.append(self.loss(y_hat, target, criterion, eval=True))
                val_loss = torch.stack(val_loss).mean()
                self.writer.add_scalar(
                    "loss/validation", val_loss, ep+1)
                pbar.set_postfix(
                    {"loss": f"{train_loss:.3f}", "acc": f"{val_loss:.3f}"})

                early_stopping_monitor.step(train_loss)
                if early_stopping_monitor.num_bad_epochs >= early_stopping_monitor.patience:
                    print("Early Stopping")
                    break
        self.ep = end
        print("----------- Training finished -----------")
        torch.cuda.empty_cache()

    @torch.no_grad()
    def predict(self, data: torch.tensor):
        return self.model(data)

    def inference(self, testset):
        self.model.eval()
        testset = self.preprocessing(testset)
        loader = DataLoader(dataset=testset,
                            batch_size=len(testset))
        x = next(iter(loader))[0]
        y = next(iter(loader))[1]
        return self.predict(x).cpu(), y.cpu()
    def loss(self, y_hat, y, criterion, eval):
        if self.target_transform:
            y = self.target_transform(y_hat, y)
        if eval and self.eval_metrics:
            return self.eval_metrics(y_hat, y)
        else:
            return criterion(y_hat, y)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __del__(self):
        self.writer.close()

    @property
    def weight(self):
        return "\n".join(map(lambda x: str(x), self.model.named_parameters()))

    def parameters(self):
        return self.model.parameters()

    def model_overview(self, model):
        from torchinfo import summary
        from itertools import count
        s = summary(model, row_settings=("var_names",))
        c = count()
        table = []
        params_num = 0
        for x in s.summary_list:
            if x.depth == 1:
                table.append([next(c), x.var_name, type(
                    x.module).__name__, x.num_params])
                params_num += x.num_params
        if s.trainable_params - params_num > 0:
            table.append([next(c), "other", "---",
                         s.trainable_params - params_num])
        table.append(["Total", "", "", s.total_params])
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        size_all_mb = (param_size + buffer_size) / 1024**2
        table.append(["Size (MB)", "", "", f"{size_all_mb:.3f}"])
        print(tabulate(table, headers=["", "Name",
                                       "Type", "Params"], tablefmt="psql", numalign="right"))
    def save(self, name):
        torch.save(self.model.state_dict(), name)
    def load(self, name):
        self.model.load_state_dict(torch.load(name))

def stratified_sampling(dataset: Dataset, train_samples_per_class: int):
    import collections
    train_indices = []
    val_indices = []
    target_counter = collections.Counter()
    for idx, (data, target) in enumerate(dataset):
        target_counter[target] += 1
        if target_counter[target] <= train_samples_per_class:
            train_indices.append(idx)
        else:
            val_indices.append(idx)
    train_dataset = Subset(dataset, train_indices)
    train_dataset = TensorDataset(torch.stack([x[0]
                                               for x in train_dataset]), torch.cat([torch.tensor([x[1]]) for x in train_dataset]))
    val_dataset = Subset(dataset, val_indices)
    val_dataset = TensorDataset(torch.stack([x[0]
                                             for x in val_dataset]), torch.cat([torch.tensor([x[1]]) for x in val_dataset]))
    return train_dataset, val_dataset


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
    T=np.zeros((len(g1b),len(g1b[0]),len(g1b[0])))
    for idx, (g1, g2) in enumerate(zip(torch.as_tensor(g1b), torch.as_tensor(g2b))):
        for i, ix in enumerate(g1):
            for j, jx in enumerate(g2):
                T[idx][i][j] = torch.square(
                ix-jx).sum()
        row_ind, col_ind = linear_sum_assignment(T[idx])
        g2b[idx] = g2b[idx][col_ind]

    return g2b

class Datasetbehaviour():
    def __init__(self, creater: callable, *args):
        self.dataid = args[-1]
        # self.shapeid = args[-1]
        key = inspect.getsource(creater)+str(args)
        filepath = self.__get_filepath(key)
        try:
            obj = pickle.load(open(filepath, 'rb'))
            print(f"[dataset loaded] -- {filepath}")
            self.dataset = obj.dataset
        except FileNotFoundError:
            creater(*(args[:-1]))
            dataset = getattr(self, self.dataid)
            self.dataset = list(map(list, zip(*dataset)))
            # self.to_tensor(self.dataset, getattr(self, self.shapeid))
            # dataset = (torch.stack([torch.tensor(np.asarray(x)) for x in dataset[0]]), torch.stack(
            #     [torch.tensor(np.asarray(x)) for x in dataset[1]]))
            # self.dataset = TensorDataset(dataset[0], dataset[1])
            pickle.dump(self, open(filepath, 'wb'))
            print("[dataset creation]")

    def __get_filepath(self, key: str):
        parent = Path("custom_datasets")
        parent.mkdir(parents=True, exist_ok=True)
        class_name = type(self).__name__
        id = hashlib.sha256(key.encode("utf-8")).hexdigest()
        file = Path(class_name + "_" + id + ".pkl")
        return parent / file

    def __getitem__(self, idx):
        return self.dataset[0][idx], self.dataset[1][idx]
    def __len__(self):
        return len(self.dataset[0])

    def save_params(self):
        frame = inspect.currentframe()
        args, _, _, values = inspect.getargvalues(frame)
        print(args)
        arg_values = {arg: values[arg] for arg in set(args) - set(['self'])}
        for arg in arg_values:
            setattr(self, arg, arg_values[arg])
    def to_tensor(self, dataset, shape=(1,1)):
        for i in range(len(dataset[0])):
            for j in range(len(shape)):
                if shape[j] > 1:
                    for k in range(shape[j]):
                        dataset[j][i][k] = torch.tensor(
                            np.asarray(dataset[j][i][k]))
                else:
                    dataset[j][i] = torch.tensor(np.asarray(dataset[j][i]))
    def union(self, instance):
        self.dataset[0] += instance.dataset[0]
        self.dataset[1] += instance.dataset[1]
        return self


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
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
        return self.pe[:, :x.size(1)]
def plotly_to_array(fig):
    image_bytes = fig.to_image(format="jpg")
    return np.asarray(Image.open(io.BytesIO(image_bytes)))
def seaborn_to_array(fig):
    return np.asarray(fig.get_figure().canvas.buffer_rgba())
def visualize_attentions(maps):
    map_num = len(maps[0]) - 1
    fig = make_subplots(rows=1, cols=map_num+1,horizontal_spacing=0)

    for i in range(len(maps)):
        for j in range(1, map_num+1):
            fig.add_heatmap(z=maps[i][j],visible=False,row=1,col=j,showscale=True if j==1 else False)
        fig.add_image(z=maps[i][0],visible=False,row=1, col=map_num+1)
    for i in range(map_num+1):
        fig.data[i].visible = True
    steps = []
    for i in range(0, len(fig.data), map_num+1):
        step = dict(
            method="update",
            args=[{"visible": [False] * len(fig.data)}],
        )
        for j in range(map_num+1):
            step["args"][0]["visible"][i+j] = True
        steps.append(step)
    sliders = [dict(
        active=10,
        currentvalue={"prefix": "index: "},
        pad={"t": 50},
        steps=steps
    )]
    for i in range(map_num+1):
        fig.update_xaxes(showgrid=False,zeroline=False,visible=False)
    for i in range(map_num+1):
        fig.update_yaxes(row=1, col=i+1, autorange="reversed", scaleanchor=f"x{i+1}", scaleratio=1,showgrid=False,zeroline=False,visible=False)
    fig.update_layout(
        sliders=sliders,
        plot_bgcolor='white',
        # autosize = False
    )
    fig.show()