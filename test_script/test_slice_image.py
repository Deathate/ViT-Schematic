# %%
import sys

sys.path.append("..")

import config
from main import *
from Model import *

# img = cv2.imread("dataset/images/circuit1001.jpg")
# img = padding(img, 100)
# h, w, c = img.shape
# p = []
# for i in range(0, h, 100):
#     for j in range(0, w, 100):
#         s = torch.tensor(img[i : i + 100, j : j + 100])
#         s = s.permute(2, 0, 1)
#         p.append(s)
# p = torch.stack(p)
# fimg = make_grid(p, nrow=w // 100)
# plot_images(fimg, 500)


class TestWindowed(Datasetbehaviour):

    def __init__(self, filepath, pad):
        img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
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


with torch.no_grad():
    img_names = (
        "218 82 850 1807 260 50001 50038 50119 50207 50799 42852 33748 8203 7735 7578 6826 6640 5841"
    ).split()
    img_names = ("24023 24167 24348").split()
    # img_names = ("24023").split()
    img_names = img_names[:5]
    slice_size = 50
    # img_names = ["5841"]
    for img_name_id in range(len(img_names)):
        path = "../dataset_100000/images/circuit" + img_names[img_name_id] + ".png"
        # path = "dataset_50x50/data_distribution_50/w_mask_w_line/images/circuit" + img_names[img_name_id] + ".png"
        Datasetbehaviour.RESET = True
        # path = "dataset/images/circuit1001.jpg"
        dataset = TestWindowed(path, slice_size)
        model = Model(
            xtransform=xtransform,
        )
        # pretrained_path = "../runs/FormalDatasetWindowed/0806_00-55-50__decoder/best.pth"
        # pretrained_path = "runs/FormalDatasetWindowed/0804_09-33-17__decoder/best.pth"
        # pretrained_path = "complete/0725_11-57-10__decoder/best.pth"
        # pretrained_path = "../runs/FormalDatasetWindowed/0813_13-03-06/best.pth"
        setting = (
            "../runs/FormalDatasetWindowed/0827_01-07-20/best.pth",
            True,
            True,
        )  # with connection, trained with full label, unbalanced loss
        setting = (
            "../runs/FormalDatasetWindowed/0827_09-05-16/best.pth",
            True,
            True,
        )  # with connection, trained with full label, balanced loss
        setting = (
            "../runs/FormalDatasetWindowed/0827_15-32-53/best.pth",
            True,
            True,
        )  # with connection, trained with partial label, balanced loss
        setting = (
            "../runs/FormalDatasetWindowed/0828_09-21-55/best.pth",
            True,
            False,
        )  # with connection, trained with partial label, balanced loss, without relation token
        setting = (
            "../runs/FormalDatasetWindowed/0828_14-18-53/best.pth",
            True,
            False,
        )  # with connection, trained with partial label, balanced loss, without relation token, fixed detection
        setting = (
            "../runs/FormalDatasetWindowed/0823_08-34-43/best.pth",
            False,
            True,
        )  # without connection
        setting = (
            "../runs/FormalDatasetWindowed/0829_11-25-45/best.pth",
            True,
            True,
        )  # xxx, with connection, trained with partial label, unbalanced loss, without relation token, free detection
        setting = (
            "../runs/FormalDatasetWindowed/0829_16-21-47/best.pth",
            True,
            True,
        )  # with connection, trained with partial label, unbalanced loss, with relation token, free detection
        setting = (
            "../runs/FormalDatasetWindowed/0829_01-47-38/best.pth",
            True,
            False,
        )  # with connection, trained with partial label, unbalanced loss, without relation token, fixed detection
        setting = (
            "../runs/FormalDatasetWindowed/0901_07-45-41/best.pth",
            False,
            False,
        )  # moco
        pretrained_path, DRAW_CONNECTION, enable_relation_token = setting
        model.fit(
            create_model(relation_token=enable_relation_token),
            pretrained_path=pretrained_path,
        )
        result = model.inference(dataset, verbose=False)
        image_set = []
        for i in range(len(result)):
            latent = result[i]
            image = dataset[i][0][:, :, :3].copy()
            h, w = image.shape[:2]
            channel = dataset[i][0][:, :, 3:]
            image_without_mask = image[np.repeat((channel > 240), 3, axis=-1)]
            # if image.sum() / 255 / slice_size**2 / 3 < 0.99 and channel.max() > 220:
            if image_without_mask.size != 0 and image_without_mask.mean() / 255 <= 0.99:
                latents = latent[: config.NUM_RESULT]
                box = model.model.box_head(latents)[:, :2]
                box_filter = (box[:, 0] >= 0) & (box[:, 1] >= 0)
                box = box[box_filter]
                latents = latents[box_filter]
                box_double_check = []
                r = 3
                for x, y in box:
                    x, y = int(x * w), int((1 - y) * h)
                    value = image[max(y - r, 0) : y + r, max(0, x - r) : x + r]
                    if value.mean() < 255:
                        box_double_check.append(True)
                    else:
                        box_double_check.append(False)
                box = box[box_double_check]
                latents = latents[box_double_check]
                image = draw_point(image, box, width=3, color=(255, 0, 0))
                if DRAW_CONNECTION and len(latents) > 1:
                    joint_token = []
                    for a, b in itertools.combinations(latents, 2):
                        if enable_relation_token:
                            joint_token.append(torch.cat((a, b, latent[-1])))
                        else:
                            joint_token.append(torch.cat((a, b)))
                    joint_token = torch.stack(joint_token)
                    relations = model.model.relation_head_wrapper(joint_token)
                    for (a, b), relation in zip(
                        itertools.combinations(list(range(len(latents))), 2), relations
                    ):
                        if F.sigmoid(relation) > 0.5:
                            p1, p2 = box[a], box[b]
                            p1 = p1.cpu()
                            p2 = p2.cpu()
                            image = draw_line(image, [(p1, p2)], thickness=2)

            image = np.concatenate((image, channel), axis=-1)
            image_set.append(image)
        fimg = create_grid(
            image_set,
            nrow=math.ceil(cv2.imread(path).shape[1] / slice_size),
            padding=5,
            pad_value=127,
        )
        plot_images(fimg, 800)
        save_path = "tmp/" + "circuit" + str(img_names[img_name_id]) + ".png"
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), fimg)
#%%