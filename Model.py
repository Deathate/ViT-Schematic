from utility import *

# snippet for importing custom modules when under folder
# sys.path.append(".." if ("ipykernel" in sys.modules) else ".")


def select_gpu_with_most_free_memory(log2console=True):
    import pynvml

    pynvml.nvmlInit()
    deviceCount = pynvml.nvmlDeviceGetCount()
    if log2console:
        print("#" * 50)
        print(f"GPU Available: {torch.cuda.is_available()}")
        print(f"CUDA_VISIBLE_DEVICES: {deviceCount}")
    memory = 0
    device = 0
    for i in range(deviceCount):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        if log2console:
            print("- DEVICE:", i)
            print(f"  TOTAL: {int(info.total / 1024**2)}, FREE: {int(info.free / 1024**2)}")
        if info.free > memory:
            memory = info.free
            device = i
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
    if log2console:
        print(f"Using GPU: [{device}]")
        print("#" * 50)


rng = np.random.default_rng()


def set_seed(seed, deterministic):
    if deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
        torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class DataCell(typing.NamedTuple):
    input: object
    output: object
    meta: dict = None


class Datasetbehaviour:
    def __init__(
        self, size, creater: abc.Callable, always_reset=False, log2console=True, *args, **kwargs
    ):
        self.size = size
        self.creater = lambda: creater(*args, **kwargs)
        self.reset = Datasetbehaviour.RESET
        self.log2console = log2console
        if not always_reset:
            source = inspect.getsource(type(self))
            source = re.sub(r"#.+\n", "", source)
            key = source + str(args) + str(kwargs) + str(size)
            filepath = self.__get_filepath(key)
            self.print("** [Create dataset]")
            self.print("[Identify file path]")
            self.print(textwrap.fill(str(filepath), 70, initial_indent=" " * 2))
            self.filepath = str(filepath)
        self.__dataset = None
        self.always_reset = always_reset

    def creater_wrapper(self):
        x = self.creater()
        if x is None:
            return None
        elif len(x) == 2:
            return x[0], x[1], None
        elif len(x) == 3:
            return x
        else:
            raise ValueError("Invalid return value")

    def __get_filepath(self, key: str):
        class_name = type(self).__name__
        parent = Path("custom_datasets") / Path(class_name)
        parent.mkdir(parents=True, exist_ok=True)
        id = hashlib.sha256(key.encode("utf-8")).hexdigest()
        file = Path(class_name + "_" + id + ".pkl")
        return parent / file

    def __load(self):
        self.print("[Load dataset]")
        self.print("- New:", Datasetbehaviour.RESET or self.always_reset)
        self.print("- Multiple Processes:", Datasetbehaviour.MP)
        if not self.always_reset and Path(self.filepath).exists() and (not Datasetbehaviour.RESET):
            self.print("** [cache found]")
            self.print(textwrap.fill(self.filepath, 70, initial_indent=" " * 4))
            self.__dataset = pickle.load(open(self.filepath, "rb"))
        else:
            if not Datasetbehaviour.MP:
                dataset = []
                for _ in tqdm(range(self.size), disable=not self.log2console):
                    dataset.append(self.creater_wrapper())
            else:
                dataset = p_tqdm.p_umap(
                    lambda _: self.creater_wrapper(),
                    range(self.size),
                    num_cpus=os.cpu_count() - 4,
                )
            self.__dataset = np.array([x for x in dataset if x is not None], dtype=object)
            pickle.dump(self.__dataset, open(self.filepath, "wb"))
        self.print("--- [Loading done] ---\n")
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
        if self.__dataset is None:
            self.__load()
        return self.__dataset

    def union_dataset(self, instance):
        if self.__dataset is None:
            self.__load()
        if len(self.__dataset) == 0:
            self.__dataset = instance.dataset()
        else:
            self.__dataset = np.concatenate([self.__dataset, instance.dataset()], axis=0)
        return self

    def reset():
        Datasetbehaviour.MP = False
        Datasetbehaviour.RESET = False

    def view(self):
        pprint(self[0])

    def print(self, *args):
        if self.log2console:
            print(*args)


Datasetbehaviour.reset()


def cudalization(x):
    if isinstance(x, torch.Tensor):
        return x.cuda(non_blocking=True)
    else:
        return [cudalization(y) for y in x]


@dataclass
class MetaData:
    data: list[DataCell]
    model: nn.Module
    epoch: int
    mode: str


class Model:
    def __init__(
        self,
        data: Datasetbehaviour = None,
        batch_size=64,
        xtransform=None,
        ytransform=None,
        validation_split=0.1,
        shuffle=False,
        amp=True,
        cudalize=True,
        use_cache=True,
        memory_fraction=1,
        eval=False,
        log2console=True,
    ):
        if memory_fraction < 1:
            torch.cuda.set_per_process_memory_fraction(memory_fraction)
        self.name = type(data).__name__
        self.xtransform = xtransform
        self.ytransform = ytransform
        if self.xtransform is None:
            self.xtransform = lambda x: torch.tensor(x).float().cuda()
        if self.ytransform is None:
            self.ytransform = lambda x: torch.tensor(x).float().cuda()
        if torch.cuda.device_count() > 1:
            select_gpu_with_most_free_memory()
        # torch.set_default_device('cuda')
        self.cudalize = cudalize
        self.data = data
        if not eval and data:
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
        self.meta: MetaData = None
        self.log2console = log2console

    def print(self, *args):
        if self.log2console:
            print(*args)

    def tensorboard_setting(self):
        if self.writer:
            self.writer.close()
        self.writer = SummaryWriter(comment="", log_dir=self.log_dir)
        # layout = {
        #     "metrics": {
        #         "Loss": ["Multiline", ["Loss/train"]],
        #         "Loss": ["Multiline", ["Loss/validation"]],
        #         "Acc": ["Multiline", ["Acc/train"]],
        #         "Acc": ["Multiline", ["Acc/validation"]],
        #         "Learning Rate": ["Multiline", ["lr"]],
        #     },
        # }
        # self.writer.add_custom_scalars(layout)

    def preprocessing(self, data: Datasetbehaviour, use_cache, cudalize):
        self.print("[data preprocessing]")
        load_from_cache = use_cache and not data.reset and not data.always_reset
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
            self.print("** [cache found]")
            result = pickle.load(open(filepath, "rb"))
        else:
            try:
                result = [
                    [i, self.xtransform(x[0]), self.ytransform(x[1])]
                    for i, x in enumerate(tqdm(data, disable=not self.log2console))
                ]
            except Exception as e:
                self.print("Error in transformation")
                raise (e)

            if load_from_cache:
                filepath.parent.mkdir(parents=True, exist_ok=True)
                pickle.dump(result, open(filepath, "wb"))

        if cudalize:
            self.print("*cudalized")
            for r in tqdm(result, disable=not self.log2console):
                r[1] = cudalization(r[1])
                r[2] = cudalization(r[2])
        self.print("--- [finish preprocessing] ---\n")
        return result

    def fit(
        self,
        model,
        criterion=None,
        optimizer=None,
        epochs=1,
        max_epochs=1e7,
        start_epoch=None,
        compile=False,
        target_transform=lambda y_hat, y: y,
        early_stopping=False,
        eval_metrics=None,
        training_epoch_end=None,
        pretrained_path="",
        keep=True,
        backprop_freq=1,
        device_ids=[0],
        keep_epoch=True,
        keep_optimizer=True,
        config=None,
        upload=False,
    ):
        now = datetime.datetime.now()
        self.log_dir = Path(f"runs/{self.name}/" + now.strftime("%m%d_%H-%M-%S") + "/")
        self.run = None
        self.upload = upload
        if config is not None:
            config["Created"] = datetime.datetime.now()
            config["Model"] = model.__class__.__name__
            config["Log_dir"] = self.log_dir
            machine_name = platform.node()
            wandb.require("core")
            wandb.login(key="dc1e94a79b4faf6ca55ddce9640f3568ef5081a5")
            self.run = wandb.init(
                # set the wandb project where this run will be logged
                project=self.name,
                name=f"{machine_name}",
                # track hyperparameters and run metadata
                config=config,
                reinit=True,
            )

            self.runapi = wandb.Api().run(f"{self.run.entity}/{self.run.project}/{self.run.id}")
            self.log_interval = defaultdict(lambda: Timer())

        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, device_ids))
        backprop_freq = int(backprop_freq)
        previous_epoch = 0
        list_of_files = glob.glob(str(self.log_dir.parent) + "/*")
        scaler = torch.cuda.amp.GradScaler()
        if pretrained_path == "latest":
            if len(list_of_files) == 0:
                pretrained_path = ""
            else:
                latest_file = max(list_of_files, key=os.path.getctime)
                pretrained_path = latest_file + "/latest.pth"
                if not Path(pretrained_path).exists():
                    self.print(f"** [Pretrained model not found] - {pretrained_path}")
                    pretrained_path = ""
        if pretrained_path:
            self.print(f'** [Pretrained model loaded] - "{pretrained_path}"')
            checkpoint = torch.load(pretrained_path)
            if isinstance(checkpoint, OrderedDict):
                model.load_state_dict(checkpoint, strict=True)
                model = self.parallel(model, device_ids)
            else:
                # it is recommended to move a model to GPU before constructing an optimizer
                model.load_state_dict(checkpoint["model"], strict=False)
                model = self.parallel(model, device_ids)
                if optimizer and keep_optimizer:
                    self.print("** [Pretrained optimizer loaded]")
                    optimizer.load_state_dict(checkpoint["optimizer"])
                if checkpoint.get("epoch", False) and keep_epoch:
                    self.print("** [Pretrained epoch loaded]")
                    previous_epoch = checkpoint["epoch"] + 1
                if checkpoint.get("scaler", False) and self.amp:
                    self.print("** [Pretrained scaler loaded]")
                    scaler.load_state_dict(checkpoint["scaler"])

        else:
            model = self.parallel(model, device_ids)
        self.model = model
        # if self.run:
        #     self.run.watch(model, log="all")
        # accelerate training speed
        if compile:
            self.model = torch.compile(self.model, mode="reduce-overhead")
        if not keep or not criterion or not optimizer:
            return
        if id(model) != self.model_id:
            self.model_id = id(model)
            self.gc()
            self.tensorboard_setting()
            shutil.copy(inspect.getfile(self.model.__class__), Path(self.writer.log_dir))
            self.model_overview(self.model)
            self.print(f"Model: {self.model.__class__.__name__}, ID:{self.model_id}")
        if self.interrupt:
            return
        try:
            best_train_loss = 1e5
            best_val_loss = 1e5
            best_acc = defaultdict(lambda: -1e5)
            best_acc_val = defaultdict(lambda: -1e5)

            self.print("----------- Training started -----------")
            start_time = time.time()
            if start_epoch is not None:
                self.ep = start_epoch
            start = self.ep
            end = min(max_epochs, start + previous_epoch + epochs)
            if early_stopping:
                early_stopping_monitor = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, patience=30
                )
            # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, min_lr=1e-6)

            self.model.train()

            def create_meta(seq, ep, mode) -> MetaData:
                return MetaData(
                    data=[DataCell(*self.data[s]) for s in seq],
                    model=self.model if len(device_ids) == 1 else self.model.module,
                    epoch=ep,
                    mode=mode,
                )

            for ep in range(start + previous_epoch, min(end, start + previous_epoch + epochs)):

                with tqdm(
                    total=len(self.train_loader) // backprop_freq,
                    bar_format="{desc}{n_fmt}/{total_fmt}|{bar}| - {elapsed}s{postfix}",
                    ncols=0,
                    disable=not self.log2console,
                ) as pbar:
                    pbar.set_description(f"Epoch {ep+1}/{end} ({max_epochs})")
                    train_loss = []
                    pbar.set_postfix({"TLoss": "---", "VLoss": "---"})
                    accs = defaultdict(list)

                    try:
                        self.model.train()
                        for batch_num, (seq, data, target) in enumerate(self.train_loader):
                            if not self.cudalize:
                                data = cudalization(data)
                                target = cudalization(target)
                            with torch.autocast("cuda", enabled=self.amp):
                                y_hat = self.model_forward(data, target)
                                self.meta = create_meta(seq, ep, "train")
                                loss, acc_data = self.loss(
                                    y_hat,
                                    target,
                                    criterion,
                                    eval=True,
                                    target_transform=target_transform,
                                    eval_metrics=eval_metrics,
                                )
                            for acc in acc_data:
                                accs[acc].append(acc_data[acc])
                            # loss = loss / backprop_freq
                            scaler.scale(loss / backprop_freq).backward()
                            train_loss.append(loss.item())

                            if (batch_num + 1) % backprop_freq == 0:
                                scaler.step(optimizer)
                                scaler.update()
                                optimizer.zero_grad()
                                # if ep == 0:
                                #     from torchviz import make_dot
                                #     graph = make_dot(y_hat, params=dict(model.named_parameters()))
                                #     graph.render(Path(self.writer.log_dir) /
                                #                  "model_graph", format="png")

                                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                                pbar.set_postfix(
                                    {
                                        "TLoss": f"{np.mean(train_loss[-backprop_freq:]):.3E}",
                                        "VLoss": "---",
                                    }
                                )
                                pbar.update(1)
                        if training_epoch_end:
                            training_epoch_end()
                    except Exception as e:
                        self.print("!!!-------- Error captured --------!!!")
                        try:
                            error = tabulate(
                                [
                                    ["y_hat", y_hat.dtype, list(y_hat.shape)],
                                    ["target", target.dtype, list(target.shape)],
                                ],
                                headers=["", "dtype", "shape"],
                                tablefmt="psql",
                            )
                            self.print(error)
                        except:
                            pass
                        self.print("Removing log directory")
                        shutil.rmtree(self.log_dir)
                        raise e

                    # calculate validation loss
                    self.model.eval()
                    with torch.no_grad():
                        val_loss = []
                        accs_val = defaultdict(list)
                        for seq, data, target in self.test_loader:
                            if not self.cudalize:
                                data = cudalization(data)
                                target = cudalization(target)
                            y_hat = self.predict(data, target)
                            self.meta = create_meta(seq, ep, mode="val")
                            loss, acc_data = self.loss(
                                y_hat,
                                target,
                                criterion,
                                eval=True,
                                target_transform=target_transform,
                                eval_metrics=eval_metrics,
                            )
                            val_loss.append(loss.item())
                            for acc in acc_data:
                                accs_val[acc].append(acc_data[acc])

                    mean_train_loss = np.mean(train_loss)
                    for acc, item in accs.items():
                        value = np.mean(item)
                        accs[acc] = value
                        best_acc[acc] = max(value, best_acc[acc])
                    mean_val_loss = np.mean(val_loss)
                    for acc, item in accs_val.items():
                        value = np.mean(item)
                        accs_val[acc] = value
                        best_acc_val[acc] = max(value, best_acc_val[acc])
                    best_train_loss = min(mean_train_loss, best_train_loss)
                    best_val_loss = min(mean_val_loss, best_val_loss)

                    self.writer.add_scalar("Loss/train", mean_train_loss, ep + 1)
                    self.writer.add_scalar("Loss/val", mean_val_loss, ep + 1)
                    for acc in accs:
                        self.writer.add_scalar(f"Accuracy/train/{acc}", accs[acc], ep + 1)
                    for acc in accs_val:
                        self.writer.add_scalar(f"Accuracy/val/{acc}", accs_val[acc], ep + 1)
                    self.writer.add_scalar("Loss/best/train", best_train_loss, ep + 1)
                    self.writer.add_scalar("Loss/best/val", best_val_loss, ep + 1)
                    for acc in best_acc:
                        self.writer.add_scalar(f"Accuracy/best/train/{acc}", best_acc[acc], ep + 1)
                    for acc in best_acc_val:
                        self.writer.add_scalar(
                            f"Accuracy/best/val/{acc}", best_acc_val[acc], ep + 1
                        )
                    # self.writer.add_scalar("lr", optimizer.param_groups[0]["lr"], ep + 1)
                    if self.run is not None:
                        self.run.log(
                            {
                                "epoch": ep + 1,
                                "Loss/train": mean_train_loss,
                                "Loss/val": mean_val_loss,
                                "Loss/best/train": best_train_loss,
                                "Loss/best/val": best_val_loss,
                                **{f"Accuracy/train/{acc}": accs[acc] for acc in accs},
                                **{f"Accuracy/val/{acc}": accs_val[acc] for acc in accs_val},
                                **{f"Accuracy/best/train/{acc}": best_acc[acc] for acc in best_acc},
                                **{
                                    f"Accuracy/best/val/{acc}": best_acc_val[acc]
                                    for acc in best_acc_val
                                },
                            },
                            step=ep + 1,
                        )
                    pbar.set_postfix(
                        {
                            "TLoss": f"{mean_train_loss:.3E}",
                            "VLoss": f"{mean_val_loss:.3E}",
                        }
                    )
                    if mean_val_loss == best_val_loss:
                        saved_path = Path(self.log_dir) / "best.pth"
                        self.save_model(saved_path, model, device_ids, optimizer, scaler, ep, False)
                        # print(f"Best model saved: {saved_path}")
                    self.save_model(
                        Path(self.log_dir) / "latest.pth",
                        model,
                        device_ids,
                        optimizer,
                        scaler,
                        ep,
                        False,
                    )

                    # scheduler.step(val_loss)
                    if early_stopping:
                        early_stopping_monitor.step(train_loss)
                        if early_stopping_monitor.num_bad_epochs >= early_stopping_monitor.patience:
                            print("Early Stopping")
                            break
            self.ep = ep + 1
        except KeyboardInterrupt:
            self.print("Keyboard interrupt received.")
            self.interrupt = True
            # if ep - previous_epoch < 20:
            #     print("Removing log directory")
            #     shutil.rmtree(self.writer.log_dir)

        end_time = time.time()
        self.print(f"Elapsed time: {end_time - start_time + self.total_time:.3f} seconds")
        self.total_time += end_time - start_time
        self.print("----------- Training finished -----------")
        self.save_model(
            Path(self.log_dir) / "latest.pth",
            model,
            device_ids,
            optimizer,
            scaler,
            ep,
            True,
        )
        self.save_model(
            Path(self.log_dir) / "best.pth",
            model,
            device_ids,
            optimizer,
            scaler,
            ep,
            True,
        )

    def model_forward(self, x, y):
        if self.model.forward.__code__.co_argcount == 2:
            return self.model(x)
        else:
            return self.model(x, y)

    def save_model(self, path, model, device_ids, optimizer, scaler, ep, force):
        torch.save(
            {
                "model": (
                    model.state_dict() if len(device_ids) == 1 else model.module.state_dict()
                ),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": ep,
            },
            path,
        )
        if self.run is not None and self.upload:
            # artifact_name = "checkpoint"
            # try:
            #     model_artifact = wandb.use_artifact(artifact_name, type="model")
            # except:
            #     model_artifact = wandb.Artifact(artifact_name, type="model")
            # self.run.log_artifact(model_artifact)
            # self.run.save(str(path), policy="now")
            name = f"runs-{self.run.id}-{Path(path).name}"
            if self.log_interval[name].elapsed() > 3600 or force:
                # self.run.log_model(path, name=name).wait()
                self.log_interval[name].reset()
                model_artifact = wandb.Artifact(name, type="model")
                model_artifact.add_file(path)
                self.run.log_artifact(model_artifact).wait()
                time.sleep(1)
                for artifact in self.runapi.logged_artifacts():
                    if artifact.type == "model" and ("latest" not in artifact.aliases):
                        artifact.delete()

    def parallel(self, model, device_ids):
        if len(device_ids) > 1:
            return nn.DataParallel(model, device_ids=list(range(len(device_ids)))).cuda()
        else:
            return model.cuda()

    @torch.no_grad()
    def predict(self, data: torch.tensor, target: torch.tensor):
        return self.model_forward(data, target)

    # def __call__(self, data, target):
    #     return self.predict(self.xtransform(data), self.ytransform(target).unsqueeze(0).unsqueeze(0).cuda())

    @torch.no_grad()
    def inference(self, testset, preprocessing=False, verbose=True):
        assert isinstance(testset, Datasetbehaviour), "testset must be Datasetbehaviour"
        # print("[inference]")
        self.model.eval()
        if not preprocessing:
            testset = self.preprocessing(testset, True, True)
        loader = DataLoader(dataset=testset, batch_size=len(testset))
        x = next(iter(loader))[1]
        y = next(iter(loader))[2]
        prediction = self.predict(x, y)
        if isinstance(prediction, tuple):
            prediction = prediction[0]
        if verbose:
            ret = list(zip(x, prediction, y))
        else:
            ret = prediction
        return ret

    # def __call__(self, x, y) -> gc.Any:
    #     x = self.xtransform(x)
    #     y = self.ytransform(y)
    #     return self.model(x, y)

    def loss(self, y_hat, y, criterion, eval, target_transform, eval_metrics):
        try:
            y = target_transform(y_hat, y)
        except Exception as e:
            if self.amp and isinstance(e, ValueError):
                pass
            else:
                raise (e)
            # pass
        if eval and eval_metrics:
            return eval_metrics(criterion, y_hat, y)
        else:
            result = criterion(y_hat, y)
            if isinstance(result, tuple):
                return result[0], {}
            else:
                return result, {}

    @property
    def weight(self):
        return "\n".join(map(lambda x: str(x), self.model.named_parameters()))

    def model_overview(self, model):
        with HiddenPrints():
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
        self.print(
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
        self.print(f"Garbage collector: collected {collected} objects.")
        torch.cuda.empty_cache()
        self.ep = 0
        self.interrupt = False
        self.total_time = 0

    def device(self):
        return next(self.model.parameters()).device


# def loss_func(criterion):
#     return lambda y_hat, y, meta: criterion(y_hat, y)


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
