# https://ncps.readthedocs.io/en/latest/examples/atari_bc.html
from shapely import LineString, box, intersects

import main_line_config as config
from Model import *

# from vit import Transformer


Datasetbehaviour.RESET = True


bad_image_dir = Path("bad_images")
shutil.rmtree(bad_image_dir, ignore_errors=True)
bad_image_dir.mkdir(exist_ok=True)
good_image_dir = Path("good_images")
shutil.rmtree(good_image_dir, ignore_errors=True)
good_image_dir.mkdir(exist_ok=True)
num_bad_image = 0
num_good_image = 0


class FormalDatasetWindowedLinePair(Datasetbehaviour):
    def __init__(self, size, dataset_source):
        self.dataset_source = dataset_source
        self.dataset_folder = Path(self.dataset_source) / Path("pkl")
        self.dataset_list = list(self.dataset_folder.iterdir())
        self.dataset_list.sort()
        # random.shuffle(self.dataset_list)
        if size == -1:
            size = len(self.dataset_list)
        self.dataset_list = self.dataset_list[:size]
        self.i = 0
        self.max_num_points = 0
        super().__init__(size, self.__create)

    def __create(self):
        path = self.dataset_list[self.i]
        self.i += 1
        data = pickle.load(open(path, "rb"))
        img_path = self.dataset_source + "/images/" + Path(path).stem + ".png"
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        img_height, img_width = img.shape[:2]
        # print(path)
        line_set = list(chain.from_iterable(data.values()))

        global num_bad_image, num_good_image
        line_set = list(filter(lambda x: len(x) == 2, line_set))
        for line in line_set:
            line[0] = (line[0][0] / img_width, line[0][1] / img_height)
            line[1] = (line[1][0] / img_width, line[1][1] / img_height)
        r = 0.005
        b = box(r, r, 1 - r, 1 - r)
        line_set = list(filter(lambda x: intersects(LineString(x), b), line_set))
        if config.STYLE == "line":
            # if Path(path).stem.startswith("circuit575_4_8"):
            #     plot_images(img)
            #     print(line_set)
            #     exit()
            img[:, :, 3] = 255
            if img.mean() == 255:
                return None

            non_white_pixels_coordinates = (img != 255).nonzero()

            non_white_pixel_range = (
                non_white_pixels_coordinates[1].max()
                - non_white_pixels_coordinates[1][0].min()
                + non_white_pixels_coordinates[0].max()
                - non_white_pixels_coordinates[0].min()
            )

            if len(line_set) == 0 or non_white_pixel_range < 5:
                return None
                out_dir = bad_image_dir / str(self.i // 1000)
                out_dir.mkdir(exist_ok=True)
                out_path = out_dir / (Path(path).stem + ".png")
                cv2.imwrite(str(out_path), cv2.resize(img, (256, 256)))
                num_bad_image += 1
                if num_bad_image == 500:
                    exit()
                return None
            # else:
            #     split_dir = good_image_dir / str(num_good_image // 1000)
            #     split_dir.mkdir(exist_ok=True)
            #     out_path = str(split_dir / (Path(path).stem + ".png"))
            #     cv2.imwrite(out_path, cv2.resize(img, (256, 256)))
            #     num_good_image += 1

            # if Path(path).stem == "circuit2095_5_1":
            #     plot_images(img)
            #     print(line_set)
            #     exit()
        elif config.STYLE == "mask":
            if len(line_set) == 0:
                return None
            line_set = list(filter(lambda x: norm1(*x) > 0.05, line_set))
            # if Path(path).stem == "circuit2326_1_11":
            #     print(img_path)
            #     plot_images(draw_line(img, line_set, thickness=1))
            #     print(line_set)
            #     exit()

            # split_dir = good_image_dir / str(num_good_image // 1000)
            # split_dir.mkdir(exist_ok=True)
            # out_path = str(split_dir / (Path(path).stem + ".png"))
            # cv2.imwrite(out_path, cv2.resize(img, (256, 256)))
            # num_good_image += 1
            # if num_good_image == 500:
            #     exit()

            # opaque_mask = img[:, :, -1] != 255
            # image_without_alpha = img[:, :, :3].copy()
            # image_without_alpha[opaque_mask.nonzero()] = (50, 50, 50)
            # plot_images(img, 300)
            # exit()
        else:
            raise ValueError("STYLE must be either 'line' or 'mask'")
        if len(line_set) > self.max_num_points:
            self.max_num_points = len(line_set)
            # print(self.max_num_points)
        return img, line_set, path


result_num = 15


def xtransform(x):
    gray_channel = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.Grayscale(),
            transforms.ToDtype(torch.float32, scale=True),
        ]
    )(x[:, :, :3])
    alpha_channel = x[:, :, -1] / 255
    alpha_channel = alpha_channel[np.newaxis, :]
    joint = torch.cat((gray_channel, torch.tensor(alpha_channel)), dim=0)
    return joint


def ytransform(x):
    s = np.full((result_num, 2, 2), 0, dtype=np.float32)
    if len(x) > 0:
        s[: len(x)] = x
    return torch.tensor(s).float()


class ViT_ex(nn.Module):
    def __init__(
        self,
        patch_size,
        dim,
        depth,
        heads,
        dim_ff,
        result_num,
        dropout=0,
        channels=3,
        style="decoder",
        relation_token=True,
    ):
        super().__init__()
        self.style = style

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        # image_height, image_width = image_size, image_size

        # self.transformer = Transformer(dim, depth, heads, dim_head, dim_head, dropout=dropout)
        self.use_mamba = "mamba" in style
        self.use_encoder = "encoder" in style
        self.use_decoder = "decoder" in style
        self.use_moco = "moco" in style
        self.relation_token = relation_token

        patch_height, patch_width = patch_size, patch_size
        # assert (
        #     image_height % patch_height == 0 and image_width % patch_width == 0
        # ), "Image dimensions must be divisible by the patch size."

        patch_dim = channels * patch_height * patch_width
        self.to_patch_embedding = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = PositionalEncoding(dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(dim, heads, dim_ff, batch_first=True, dropout=dropout),
            depth,
        )

        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(dim, heads, dim_ff, batch_first=True, dropout=dropout),
            depth,
        )
        self.decoder_query = nn.Embedding(result_num, dim)

        self.box_head = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, 4)
        )

    def forward(self, x, y):
        x = x.to(torch.float32)
        x = self.to_patch_embedding(x)
        x += self.pos_embedding(x)
        x = self.transformer(x)
        tgt = repeat(self.decoder_query.weight, "d e -> n d e", n=x.shape[0])
        x = self.decoder(tgt, x)
        x = self.box_head(x)
        x = x.reshape(x.shape[0], -1, 2, 2)
        return x


def create_model(**kwargs):
    style = config.MODEL_STYLE
    depth = config.DEPTH
    head_num = config.NUM_HEADS
    dim_head = config.EMBED_DIM
    parameters = {
        "patch_size": config.PATCH_SIZE,
        "dim": head_num * dim_head,
        "depth": depth,
        "heads": head_num,
        "dim_ff": head_num * dim_head,
        "result_num": result_num,
        "channels": 2,
        "dropout": config.DROPOUT,
        "style": style,
    }
    parameters |= kwargs
    network = ViT_ex(**parameters)
    return network


def Hungarian_Order(g1b, g2b):
    indices = []

    C1 = torch.cdist(g1b[:, :, 0], g2b[:, :, 0]) + torch.cdist(g1b[:, :, 1], g2b[:, :, 1])
    C2 = torch.cdist(g1b[:, :, 0], g2b[:, :, 1]) + torch.cdist(g1b[:, :, 1], g2b[:, :, 0])
    C3 = torch.min(C1, C2).cpu().detach()

    indices = [linear_sum_assignment(c)[1] for c in C3]
    for i in range(len(indices)):
        ind = indices[i]
        g2b[i] = g2b[i][ind]
    # 32 15 2 2
    C1 = torch.abs(g1b[:, :, 0] - g2b[:, :, 0]) + torch.abs(g1b[:, :, 1] - g2b[:, :, 1])
    C1 = C1.sum(dim=2)
    C2 = torch.abs(g1b[:, :, 0] - g2b[:, :, 1]) + torch.abs(g1b[:, :, 1] - g2b[:, :, 0])
    C2 = C2.sum(dim=2)
    min_index = C1 > C2
    g2b[min_index] = g2b[min_index][:, [1, 0]]


def criterion(y_hat, y):
    Hungarian_Order(y_hat, y)
    loss_box = F.smooth_l1_loss(y_hat, y)
    return loss_box


def eval_metrics(criterion, y_hat, y):
    loss = criterion(y_hat, y)
    C = torch.cdist(y_hat[:, :, 0], y[:, :, 0]) + torch.cdist(y_hat[:, :, 1], y[:, :, 1])
    accs = sum([(c.diag() < 0.05).sum() for c in C]) / (C.size(0) * C.size(1))
    return loss, {"acc": accs.item()}


def main():
    network = create_model()
    Datasetbehaviour.MP = False
    if isinstance(config.DATASET_SIZE, list):
        datasets = []
        for a, b in zip(config.DATASET_SIZE, config.DATASET_PATH):
            Datasetbehaviour.RESET = True
            datasets.append(FormalDatasetWindowedLinePair(a, b))
            print(a, b, len(datasets[-1]))
        # exit()
        dataset_guise = datasets[0]
        for i in range(1, len(datasets)):
            dataset_guise = dataset_guise.union_dataset(datasets[i])
    else:
        dataset_guise = FormalDatasetWindowedLinePair(config.DATASET_SIZE, config.DATASET_PATH)

    if config.EVAL:
        dataset_guise.view()

    model = Model(
        dataset_guise,
        xtransform=xtransform,
        ytransform=ytransform,
        amp=True,
        batch_size=config.BATCH_SIZE,
        eval=config.EVAL,
    )

    model.fit(
        network,
        criterion,
        optim.Adam(network.parameters(), lr=config.LEARNING_RATE),
        config.EPOCHS,
        max_epochs=config.MAX_EPOCHS if hasattr(config, "MAX_EPOCHS") else float("inf"),
        pretrained_path=config.PRETRAINED_PATH,
        keep=not config.EVAL,
        backprop_freq=config.BATCH_STEP,
        device_ids=config.DEVICE_IDS,
        eval_metrics=eval_metrics,
        keep_epoch=config.KEEP_EPOCH,
        keep_optimizer=config.KEEP_OPTIMIZER,
        config=(
            (get_attr(config) if not config.EVAL else None)
            if sum(config.DATASET_SIZE) > 3000 or sum(config.DATASET_SIZE) < 0
            else None
        ),
        upload=config.UPLOAD,
    )


if __name__ == "__main__":
    main()
