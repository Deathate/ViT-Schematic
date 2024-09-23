# %%
# https://ncps.readthedocs.io/en/latest/examples/atari_bc.html
from numpy.linalg import pinv
from scipy.spatial.distance import cdist

import config_ae as config
from Model import *
from vit import Transformer


class FormalDatasetWindowedSS(Datasetbehaviour):
    def __init__(self, size, dataset_source):
        self.dataset_source = dataset_source
        self.dataset_folder = Path(self.dataset_source) / Path("pkl")
        self.dataset_list = list(self.dataset_folder.iterdir())
        random.shuffle(self.dataset_list)
        if size is None or size < 0:
            size = len(self.dataset_list)
        self.dataset_list = self.dataset_list[:size]
        self.i = 0
        self.max_num_points = 0
        super().__init__(size, self.__create)

    def __create(self):
        path = self.dataset_list[self.i]
        self.i += 1
        img = cv2.imread(
            self.dataset_source + "/images/" + Path(path).stem + ".png", cv2.IMREAD_UNCHANGED
        )
        return img, img


result_num = config.NUM_RESULT


def xtransform(x):
    gray_channel = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.Grayscale(),
            transforms.ToDtype(torch.float, scale=True),
        ]
    )(x[:, :, :3])
    alpha_channel = x[:, :, -1] / 255
    alpha_channel = alpha_channel[np.newaxis, :]
    alpha_channel = torch.tensor(alpha_channel)
    joint = torch.cat((gray_channel, alpha_channel), dim=0)
    joint = joint.to(torch.float32)
    return joint


class Autoencoder(nn.Module):
    def __init__(self, middle_dim):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(5000, middle_dim))
        # self.middle = nn.Sequential(
        #     nn.Linear(middle_dim, middle_dim // 2),
        #     nn.Linear(middle_dim // 2, middle_dim),
        # )
        self.decoder = nn.Sequential(nn.Linear(middle_dim, 5000))

    def forward(self, x, y):
        shape = x.shape
        x = x.view(x.size(0), -1)
        x = self.encoder(x)
        # x = self.middle(x)
        x = self.decoder(x)
        x = x.view(shape)
        return x


def create_model():
    network = Autoencoder(200)
    return network


def main():
    if config.DETERMINISTIC:
        set_seed(0, True)
    else:
        set_seed(0, False)
    network = create_model()
    Datasetbehaviour.MP = False
    if config.EVAL:
        Datasetbehaviour.RESET = True
    datasets = []
    for a, b in zip(config.DATASET_SIZE, config.DATASET_PATH):
        datasets.append(FormalDatasetWindowedSS(a, b))
    dataset_guise = datasets[0]
    for i in range(1, len(datasets)):
        dataset_guise = dataset_guise.union_dataset(datasets[i])
    if config.EVAL:
        dataset_guise.view()

    model = Model(
        "Autoencoder",
        dataset_guise,
        xtransform=xtransform,
        ytransform=xtransform,
        amp=False,
        batch_size=config.BATCH_SIZE,
        eval=config.EVAL,
    )

    def criterion(y_hat, y):
        loss = F.smooth_l1_loss(y_hat, y)
        return loss

    model.fit(
        network,
        criterion,
        # optim.SGD(network.parameters(), lr=config.LEARNING_RATE, momentum=0.99),
        optim.Adam(network.parameters(), lr=config.LEARNING_RATE),
        config.EPOCHS,
        max_epochs=config.MAX_EPOCHS if hasattr(config, "MAX_EPOCHS") else float("inf"),
        pretrained_path=config.PRETRAINED_PATH,
        keep=not config.EVAL,
        backprop_freq=config.BATCH_STEP,
        device_ids=config.DEVICE_IDS,
        keep_epoch=config.KEEP_EPOCH,
        keep_optimizer=config.KEEP_OPTIMIZER,
        config=(get_attr(config) if not config.EVAL else None),
    )


if __name__ == "__main__":
    main()
