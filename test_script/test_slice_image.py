# %%
import sys

sys.path.append("..")

from main_windowed import *
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

    def __init__(self, filepath):
        img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
        img = padding(img, 100)
        h, w, c = img.shape
        p = []
        for i in range(0, h, 100):
            for j in range(0, w, 100):
                s = img[i : i + 100, j : j + 100]
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
    img_names = ["5841"]
    for img_name_id in range(len(img_names)):
        path = "../dataset_100000/images/circuit" + img_names[img_name_id] + ".png"
        Datasetbehaviour.RESET = True
        # path = "dataset/images/circuit1001.jpg"
        dataset = TestWindowed(path)
        model = Model(
            xtransform=xtransform,
        )
        pretrained_path = "../runs/FormalDatasetWindowed/0806_00-55-50__decoder/best.pth"
        # pretrained_path = "runs/FormalDatasetWindowed/0804_09-33-17__decoder/best.pth"
        # pretrained_path = "complete/0725_11-57-10__decoder/best.pth"
        # pretrained_path = "latest"
        model.fit(
            network,
            pretrained_path=pretrained_path,
        )
        result = model.inference(dataset, verbose=False)
        image_set = []
        for i in range(len(result)):
            latent = result[i]
            image = dataset[i][0][:, :, :3].copy()

            # joint_token = []
            # for a, b in itertools.combinations(latent[: config.NUM_RESULT], 2):
            #     joint_token.append(torch.cat((a, b, latent[-1])))
            # joint_token = torch.stack(joint_token)
            # relations = model.model.relation_heads(joint_token)
            # for (a, b), relation in zip(
            #     itertools.combinations(latent[: config.NUM_RESULT], 2), relations
            # ):
            #     if F.sigmoid(relation) > 0.5:
            #         p1 = model.model.box_head(a).detach().cpu()[:2]
            #         p2 = model.model.box_head(b).detach().cpu()[:2]
            #         if all(p1 >= 0) and all(p2 >= 0):
            #             image = draw_line(image, [(p1, p2)])

            box = model.model.box_head(latent[: config.NUM_RESULT])[:, :2]
            box = box[(box[:, 0] >= 0) & (box[:, 1] >= 0)]
            image = draw_point(image, box, width=5, color=(255, 0, 0))
            image = np.concatenate((image, dataset[i][0][:, :, 3:]), axis=-1)
            image_set.append(image)
        fimg = create_grid(image_set, nrow=7, padding=0)
        plot_images(fimg, 500)
        save_path = "tmp/" + "circuit" + str(img_names[img_name_id]) + ".png"
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), fimg)
