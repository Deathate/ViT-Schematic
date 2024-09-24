from main_line import *
from Model import *


class TestWindowed(Datasetbehaviour):

    def __init__(self, filepath, interval, window_size):
        img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(filepath)
            raise FileNotFoundError
        img = padding(img, window_size)
        h, w, c = img.shape
        p = []
        for i in range(0, h, window_size // interval):
            for j in range(0, w, window_size // interval):
                s = img[i : i + window_size, j : j + window_size]
                p.append(s)
        self.dataset = p
        self.i = 0
        super().__init__(len(self.dataset), self.__create, log2console=False)

    def __create(self):
        result = self.dataset[self.i]
        self.i += 1
        return result, 0


class OneTimeWrapper(Datasetbehaviour):

    def __init__(self, img):
        self.dataset = img
        super().__init__(1, self.__create, always_reset=True, log2console=False)

    def __create(self):
        return self.dataset, 0


def legalize_line(lines):
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
        if abs(line[0, 0] - line[1, 0]) < abs(line[0, 1] - line[1, 1]):
            m = (line[0, 0] + line[1, 0]) / 2
            line[0, 0] = m
            line[1, 0] = m
        else:
            m = (line[0, 1] + line[1, 1]) / 2
            line[0, 1] = m
            line[1, 1] = m
        lines[j] = line
    new_lines = []
    for line in lines:
        if np.linalg.norm(line[0] - line[1]) > 0.02:
            new_lines.append(line)
    new_lines = np.array(new_lines)
    return new_lines


@torch.no_grad()
def analyze_connection(path, debug=False):
    model = Model(
        xtransform=xtransform,
        log2console=False,
    )
    model.fit(
        create_model(),
        pretrained_path="weights/mix_best.pth",
    )
    model_l = Model(
        xtransform=xtransform,
        log2console=False,
    )
    model_l.fit(
        create_model(),
        pretrained_path="weights/line_best.pth",
    )
    slice_size = 50
    interval = 1
    if not isinstance(path, str):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            cv2.imwrite(f.name, path)
            path = f.name
    ori_img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    num_column = math.ceil(ori_img.shape[1] / slice_size)
    num_row = math.ceil(ori_img.shape[0] / slice_size)
    # path = "dataset_50x50/data_distribution_50/w_mask_w_line/images/circuit" + img_names[img_name_id] + ".png"
    Datasetbehaviour.RESET = True

    # print(cv2.imread(path))
    # exit()
    # print(path)
    # with HiddenPrints():
    dataset = TestWindowed(path, interval, slice_size)

    def predict_mask_img(img):
        img = img.copy()
        tmp = OneTimeWrapper(img)
        result = model.inference(tmp, verbose=False)
        result = legalize_line(result[0].cpu().numpy())
        return result

    def predict_line_only_img(img):
        img = img.copy()
        img[:, :, 3] = 255
        tmp = OneTimeWrapper(img)
        result = model_l.inference(tmp, verbose=False)
        result = legalize_line(result[0].cpu().numpy())
        return result

    # plot_images(ori_img, 700)
    # plot_images(
    #     create_grid(
    #         [x[0] for x in dataset],
    #         nrow=num_column,
    #         padding=5,
    #         pad_value=127,
    #     ),
    #     img_width=700,
    # )
    # exit()
    # result = model.inference(dataset, verbose=False)
    image_set = []
    global_line = []
    global_connection = []
    # debug_image = []
    from collect_connection import build_connection

    for i in range(len(dataset)):
        image_bk = dataset[i][0][:, :, :3]
        if image_bk.shape != (slice_size, slice_size, 3):
            continue
        image = image_bk.copy()
        # plot_images(image, 600)
        mask = dataset[i][0][:, :, 3:]
        h, w = image.shape[:2]
        image_without_mask = image[:, :, :3][np.repeat((mask > 240), 3, axis=-1)]
        row_idx = i // (num_column * interval)
        col_idx = i % (num_column * interval)
        anchor = (
            col_idx * slice_size / interval,
            (num_row * interval - row_idx - interval) * slice_size / interval,
        )
        if image_without_mask.size != 0 and image_without_mask.mean() / 255 <= 0.999:
            r = int(0.025 * image.shape[0])
            slice_image = image[r:-r, r:-r]
            mode = -1
            lines = []
            if mask.mean() >= 254:
                lines = predict_line_only_img(dataset[i][0])
                mode = 1
            else:
                lines = predict_mask_img(dataset[i][0])
                mode = 2
            # if [row_idx, col_idx] == [1, 9]:
            #     # plot_images(image, 300)
            #     print(lines)
            #     print(mode)
            #     print("line only")
            #     l = predict_line_only_img(dataset[i][0])
            #     plot_images(draw_line(dataset[i][0].copy(), l, thickness=1), 300)
            #     print("mask")
            #     l = predict_mask_img(dataset[i][0])
            #     plot_images(draw_line(dataset[i][0].copy(), l, thickness=1), 300)

            #     print(mask.mean())
            #     print(slice_image.mean())
            #     plot_images(draw_line(dataset[i][0].copy(), lines, thickness=1), 300)
            #     exit()

            connection = build_connection(
                lines,
                norm1,
                similar_threshold=0.02,
                threshold=0.05,
                duplicate_threshold=1e-3,
            )
            connection = list(filter(lambda x: len(x) > 0, connection))
            for j, group in enumerate(connection):
                color = color_map(j + 1)
                # image = draw_rect(image, group, color=color, width=8)
                group = np.array(group)
                group[:, 0] = group[:, 0] * slice_size + anchor[0]
                group[:, 1] = group[:, 1] * slice_size + anchor[1]
                group[:, 0] /= ori_img.shape[1]
                group[:, 1] /= ori_img.shape[0]
                global_connection.append(group.tolist())
            if len(lines) > 0:
                image = draw_line(image, lines, thickness=1)
            for line in lines:
                line[0][0] = anchor[0] + line[0][0] * slice_size
                line[0][1] = anchor[1] + line[0][1] * slice_size
                line[1][0] = anchor[0] + line[1][0] * slice_size
                line[1][1] = anchor[1] + line[1][1] * slice_size
                line[:, 0] /= ori_img.shape[1]
                line[:, 1] /= ori_img.shape[0]
                global_line.append(line.tolist())

        # image = np.concatenate((image, mask), axis=-1)
        # if row_idx == 6 and 0 <= col_idx <= 5:
        #     debug_image.append(image)
        if row_idx % interval == 0 and col_idx % interval == 0:
            image_set.append(image)

    group_connection = build_connection(
        global_connection, norm1, similar_threshold=0, threshold=0.01, duplicate_threshold=0.01
    )
    if debug:
        plot_images(ori_img, 800)
        fimg = create_grid(
            image_set,
            nrow=num_column,
            padding=5,
            pad_value=127,
        )
        plot_images(fimg, 800)
        # exit()

        img = np.full(ori_img.shape, 255, np.uint8)
        img = draw_line(img, global_line, endpoint=True, endpoint_thickness=4)
        plot_images(img, 800)
        # plot_images(debug_image, 200)

        # for i, group in enumerate(global_connection):
        #     color = color_map(i)
        #     img = draw_rect(img, group, color=color, width=10)
        # plot_images(img, 600)
        # exit()
        img = np.full(ori_img.shape, 255, np.uint8)
        img = draw_line(img, global_line, endpoint=True, endpoint_thickness=4, color=(0, 0, 0))
        img_bk = img.copy()
        process_images = []
        for i, group in enumerate(group_connection):
            img = img_bk.copy()
            img = draw_rect(img, group, color=(192, 91, 22), width=5)
            process_images.append(img.copy())
        plot_images(process_images, img_width=800)
    for i, group in enumerate(group_connection):
        color = color_map(i)
        ori_img = draw_rect(ori_img, group, color=color, width=5)
    return group_connection, ori_img
    # save_path = "tmp/" + "circuit" + str(img_names[img_name_id]) + ".png"
    # save_path = Path(save_path)
    # save_path.parent.mkdir(parents=True, exist_ok=True)
    # cv2.imwrite(str(save_path), fimg)


if __name__ == "__main__":
    img_names = (
        "218 82 850 1807 260 50001 50038 50119 50207 50799 42852 33748 8203 7735 7578 6826 6640 5841"
    ).split()
    img_name = [img_names[6]]
    # img_names = ("24023 24167 24348").split()[-2:]
    # img_names = ("24023").split()
    path = "dataset_fullimg_mask/images/circuit16176.png"
    path = "dataset_fullimg_mask/images/circuit" + img_name[0] + ".png"
    group_connection, img = analyze_connection(cv2.imread(path, cv2.IMREAD_UNCHANGED), debug=True)
    plot_images(img, 800)
