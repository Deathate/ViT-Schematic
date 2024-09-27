import sys

sys.path.append(".")
from main_line import *
from Model import *

Datasetbehaviour.MP = False
STYLE = "line"
DATASET_PATH = [
    "data_distribution_50/w_mask_w_line",
    "data_distribution_50/wo_mask_w_line",
]
if STYLE == "line":
    DATASET_SIZE = [2000, 0]
else:
    DATASET_SIZE = [0, 2000]
datasets = []
for a, b in zip(DATASET_SIZE, DATASET_PATH):
    datasets.append(FormalDatasetWindowedLinePair(a, b))
dataset_guise = datasets[0]
for i in range(1, len(datasets)):
    dataset_guise = dataset_guise.union_dataset(datasets[i])


class OneTimeWrapper(Datasetbehaviour):

    def __init__(self, img):
        self.dataset = img
        super().__init__(1, self.__create, always_reset=True, log2console=False)

    def __create(self):
        return self.dataset, 0


dataset = dataset_guise
with torch.no_grad():
    current_dir = Path(__file__).parent.parent
    model = Model(
        xtransform=xtransform,
        log2console=False,
    )
    model.fit(
        create_model(),
        pretrained_path=current_dir / "runs/FormalDatasetWindowedLinePair/0925_01-02-38/best.pth",
    )
    model_l = Model(
        xtransform=xtransform,
        log2console=False,
    )
    model_l.fit(
        create_model(),
        pretrained_path=current_dir / "runs/FormalDatasetWindowedLinePair/0925_01-40-08/best.pth",
    )

    def predict_mask_img(img):
        tmp = OneTimeWrapper(img)
        result = model.inference(tmp, verbose=False)
        return np.arrray(result[0].cpu())

    def predict_line_only_img(img):
        img = img.copy()
        img[:, :, 3] = 255
        tmp = OneTimeWrapper(img)
        result = model_l.inference(tmp, verbose=False)
        return np.array(result[0].cpu())

    for i in range(len(dataset)):
        if STYLE == "line":
            y_hat = predict_line_only_img(dataset[i][0])
        else:
            y_hat = predict_mask_img(dataset[i][0])
        y = np.array(dataset[i][1])
        C = distance.cdist(y_hat[:, 0], y[:, 0]) + distance.cdist(y_hat[:, 1], y[:, 1])
        coords = [
            (35.0456, -85.2672),
            (35.1174, -89.9711),
            (35.9728, -83.9422),
            (36.1667, -86.7833),
        ]
        print(distance.cdist(coords, coords))
        print(y_hat[:, 0], y[:, 0])
        exit()
