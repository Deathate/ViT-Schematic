# %%
import sys

sys.path.append("..")
# %%
from scipy.spatial.distance import cdist

import main_line_config as config
from main_line import *
from Model import *


class TestWindowed(Datasetbehaviour):

    def __init__(self, filepath, pad):
        img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(filepath)
            raise FileNotFoundError
        img = padding(img, pad)
        h, w, c = img.shape
        p = []
        for i in range(0, h, pad):
            for j in range(0, w, pad):
                s = img[i : i + pad, j : j + pad]
                p.append(s)
        self.dataset = p
        self.i = 0
        super().__init__(len(self.dataset), self.__create)

    def __create(self):
        result = self.dataset[self.i]
        self.i += 1
        return result, 0


class OneTimeWrapper(Datasetbehaviour):

    def __init__(self, img):
        self.dataset = img
        super().__init__(1, self.__create, always_reset=True)

    def __create(self):
        return self.dataset, 0


with torch.no_grad():
    img_names = (
        "218 82 850 1807 260 50001 50038 50119 50207 50799 42852 33748 8203 7735 7578 6826 6640 5841"
    ).split()[:1]
    # img_names = ("24023 24167 24348").split()[-2:]
    # img_names = ("24023").split()
    img_names = img_names
    slice_size = 50
    # img_names = ["5841"]
    for img_name_id in range(len(img_names)):
        path = "../dataset_fullimg_mask/images/circuit" + img_names[img_name_id] + ".png"
        ori_img = cv2.imread(path)
        num_column = math.ceil(ori_img.shape[1] / slice_size)
        num_row = math.ceil(ori_img.shape[0] / slice_size)
        # path = "dataset_50x50/data_distribution_50/w_mask_w_line/images/circuit" + img_names[img_name_id] + ".png"
        Datasetbehaviour.RESET = True
        # path = "dataset/images/circuit1001.jpg"
        dataset = TestWindowed(path, slice_size)
        model = Model(
            xtransform=xtransform,
        )

        model.fit(
            create_model(),
            pretrained_path="../runs/FormalDatasetWindowedLinePair/0912_17-40-37/best.pth",
        )
        # model fot line only
        model_l = Model(
            xtransform=xtransform,
        )
        model_l.fit(
            create_model(),
            pretrained_path="../runs/FormalDatasetWindowedLinePair/0914_22-49-35/best.pth",
        )
        result = model.inference(dataset, verbose=False)
        image_set = []
        global_line = []
        from collect_connection import build_connection

        def predict_line_only_img(img):
            tmp = OneTimeWrapper(dataset[i][0])
            result = model_l.inference(tmp, verbose=False)
            return result

        for i in range(len(result)):
            lines = result[i].cpu().numpy()
            image = dataset[i][0][:, :, :3].copy()
            image_bk = image.copy()
            h, w = image.shape[:2]
            channel = dataset[i][0][:, :, 3:]
            image_without_mask = image[np.repeat((channel > 240), 3, axis=-1)]
            if image_without_mask.size != 0 and image_without_mask.mean() / 255 <= 0.99:
                for j, line in enumerate(lines):
                    r = 0.001
                    if line[0, 0] < r:
                        line[0, 0] = 0
                    if line[0, 1] < r:
                        line[0, 1] = 0
                    if line[1, 0] < r:
                        line[1, 0] = 0
                    if line[1, 1] < r:
                        line[1, 1] = 0
                    if line[0, 0] > 1 - r:
                        line[0, 0] = 1
                    if line[0, 1] > 1 - r:
                        line[0, 1] = 1
                    if line[1, 0] > 1 - r:
                        line[1, 0] = 1
                    if line[1, 1] > 1 - r:
                        line[1, 1] = 1
                    lines[j] = line
                lines = list(filter(lambda x: x.sum() > 0, lines))
                lines = np.array(lines)
                connection = build_connection(lines, norm1, threshold=0.08)
                connection = list(filter(lambda x: len(x) > 0, connection))
                for j, group in enumerate(connection):
                    color = color_map(j)
                    image = draw_rect(image, group, color=color, width=12)
                    # print(group)
                    # plot_images(image, 600)
                    # exit()
                row_idx = i // num_column
                col_idx = i % num_column
                # if row_idx == 6 and col_idx == 7:
                #     plot_images(image_bk, 300)
                #     plot_images(image, 300)
                #     print(lines)
                #     print(connection)
                #     # print(json.dumps(lines.tolist()))
                #     exit()
                for line in lines:
                    # image = draw_line(image, [(line[0], line[1])], thickness=2)
                    global_line.append((i // num_column, i % num_column, line))
            image = np.concatenate((image, channel), axis=-1)
            image_set.append(image)
        fimg = create_grid(
            image_set,
            nrow=num_column,
            padding=5,
            pad_value=127,
        )
        fimg = cv2.cvtColor(fimg, cv2.COLOR_BGRA2BGR)
        plot_images(ori_img, 700)
        plot_images(fimg, 700)

        # transformer_global_lines = []
        # for x, y, line in global_line:
        #     anchor = y * slice_size, (num_row - x - 1) * slice_size
        #     line[0][0] = anchor[0] + line[0][0] * slice_size
        #     line[0][1] = anchor[1] + line[0][1] * slice_size
        #     line[1][0] = anchor[0] + line[1][0] * slice_size
        #     line[1][1] = anchor[1] + line[1][1] * slice_size
        #     line[:, 0] /= ori_img.shape[1]
        #     line[:, 1] /= ori_img.shape[0]
        #     transformer_global_lines.append(line.tolist())
        # # pprint(transformer_global_lines)
        # img = np.full(ori_img.shape, 255, np.uint8)
        # img = draw_line(img, transformer_global_lines, endpoint=True, endpoint_thickness=4)
        # # pprint(transformer_global_lines)

        # group_connection = build_connection(transformer_global_lines, norm1, threshold=0.05)
        # for i, group in enumerate(group_connection):
        #     if len(group) == 0:
        #         continue
        #     color = color_map(i)
        #     img = draw_rect(img, group, color=color, width=10)
        # plot_images(img, 600)
        # pprint(transformer_global_lines)
        # print()
        # pprint(group_connection)
        # save_path = "tmp/" + "circuit" + str(img_names[img_name_id]) + ".png"
        # save_path = Path(save_path)
        # save_path.parent.mkdir(parents=True, exist_ok=True)
        # cv2.imwrite(str(save_path), fimg)
# %%
