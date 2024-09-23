# %%
import sys

sys.path.append("..")
from main_full import *
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


class TestFull(Datasetbehaviour):

    def __init__(self, filepath):
        self.img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
        super().__init__(1, self.__create)

    def __create(self):
        return self.img, 0


with torch.no_grad():
    img_names = (
        "218 82 850 1807 260 50001 50038 50119 50207 50799 42852 33748 8203 7735 7578 6826 6640 5841"
    ).split()
    # img_names = ["5841"]
    for img_name_id in range(len(img_names)):
        path = "../dataset/images/circuit" + img_names[img_name_id] + ".jpg"
        Datasetbehaviour.RESET = True
        # path = "dataset/images/circuit1001.jpg"
        dataset = TestFull(path)
        model = Model(
            xtransform=xtransform,
        )
        pretrained_path = "../nvidia_log/best.pth"
        # pretrained_path = "runs/FormalDatasetWindowed/0804_09-33-17__decoder/best.pth"
        # pretrained_path = "complete/0725_11-57-10__decoder/best.pth"
        # pretrained_path = "latest"
        model.fit(
            network,
            pretrained_path=pretrained_path,
        )
        result = model.inference(dataset, verbose=False)
        image = dataset[0][0].copy()
        latent = result[0]
        box = model.model.box_head(latent[:result_num])[:, :2]
        box = box[(box[:, 0] >= 0) & (box[:, 1] >= 0)]
        image = draw_point(image, box, width=5, color=(255, 0, 0))
        plot_images(image, 500)
        save_path = "tmp/" + "circuit" + str(img_names[img_name_id]) + ".png"
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), image)
