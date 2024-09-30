# %%
from collect_connection import build_connection
from main_line import *
from Model import *


class TestWindowed(Datasetbehaviour):

    def __init__(self, img, interval, window_size):
        img = padding(img, window_size)
        h, w, c = img.shape
        p = []
        for i in range(0, h, window_size // interval):
            for j in range(0, w, window_size // interval):
                s = img[i : i + window_size, j : j + window_size]
                p.append(s)
        self.dataset = p
        self.i = 0
        super().__init__(len(self.dataset), self.__create, always_reset=True, log2console=False)

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


current_dir = Path(__file__).parent
model = Model(
    xtransform=xtransform,
    log2console=False,
)
try:
    model.fit(
        create_model(),
        pretrained_path=current_dir / "aruns/FormalDatasetWindowedLinePair/0925_11-14-32/best.pth",
    )
except:
    model.fit(
        create_model(),
        pretrained_path=current_dir / "weights/mix_best.pth",
    )
model_l = Model(
    xtransform=xtransform,
    log2console=False,
)
try:
    model_l.fit(
        create_model(),
        pretrained_path=current_dir / "runs/FormalDatasetWindowedLinePair/0925_11-13-18/best.pth",
    )
except:
    model_l.fit(
        create_model(),
        pretrained_path=current_dir / "weights/line_best.pth",
    )


def predict_mask_img(img, threshold):
    img = img.copy()
    tmp = OneTimeWrapper(img)
    result = model.inference(tmp, verbose=False)
    result = legalize_line(result[0].cpu().numpy(), threshold)
    return result


def predict_line_only_img(img, threshold):
    img = img.copy()
    img[:, :, 3] = 255
    tmp = OneTimeWrapper(img)
    result = model_l.inference(tmp, verbose=False)
    result = legalize_line(result[0].cpu().numpy(), threshold)
    return result


# %%
def legalize_line(lines, threshold):
    for j, line in enumerate(lines):
        # r = 0.03
        # if line[0, 0] < r:
        #     line[0, 0] = 0
        # if line[0, 1] < r:
        #     line[0, 1] = 0
        # if line[1, 0] < r:
        #     line[1, 0] = 0
        # if line[1, 1] < r:
        #     line[1, 1] = 0
        # if line[0, 0] > 1 - r:
        #     line[0, 0] = 1
        # if line[0, 1] > 1 - r:
        #     line[0, 1] = 1
        # if line[1, 0] > 1 - r:
        #     line[1, 0] = 1
        # if line[1, 1] > 1 - r:
        #     line[1, 1] = 1
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
        if np.linalg.norm(line[0] - line[1]) > threshold:
            new_lines.append(line)
    new_lines = np.array(new_lines)
    return new_lines


@torch.no_grad()
@hidden_matplotlib_plots
def analyze_connection(
    path,
    min_line_length,
    local_threshold,
    global_threshold,
    remove_duplicate,
    optimal_shift,
    boundary,
    strict_match,
    strict_match_threshold,
    debug,
    debug_shift_optimization,
    debug_cell,
):
    soft_match = False
    if not debug and debug_shift_optimization:
        warnings.warn("debug_shift_optimization is disabled because debug is disabled")
        debug_shift_optimization = False
    with HiddenPrints(disable=debug), HiddenPlots(disable=debug):
        slice_size = 50
        interval = 1
        if not isinstance(path, str):
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                cv2.imwrite(f.name, path)
                path = f.name
        input_img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        ori_img = padding(input_img, slice_size)
        ori_img = specify_padding(ori_img, slice_size, slice_size, fill=255)
        shift_x, shift_y = 0, 0
        if optimal_shift:
            shift_x, shift_y = calculate_optimal_shift(
                slice_size, ori_img, boundary, debug_shift_optimization
            )
            print("optimal_shift_x:", shift_x)
            print("optimal_shift_y:", shift_y)
            ori_img = shift(ori_img, (shift_x, shift_y), fill=255)
        num_column = math.ceil(ori_img.shape[1] / slice_size)
        num_row = math.ceil(ori_img.shape[0] / slice_size)
        dataset = TestWindowed(ori_img, interval, slice_size)

        image_set = []
        global_line = []
        local_lines = {}

        for i in range(len(dataset)):
            image_bk = dataset[i][0][:, :, :3]
            image_bk_gray = cv2.cvtColor(image_bk, cv2.COLOR_BGR2GRAY)
            if image_bk.shape != (slice_size, slice_size, 3):
                continue
            image = image_bk.copy()
            mask = dataset[i][0][:, :, 3:]
            image_without_mask = image[:, :, :3][np.repeat((mask > 240), 3, axis=-1)]
            row_idx = i // (num_column * interval)
            col_idx = i % (num_column * interval)
            anchor = (
                col_idx * slice_size / interval,
                (num_row * interval - row_idx - interval) * slice_size / interval,
            )
            if image_without_mask.size != 0 and image_without_mask.mean() / 255 <= 0.999:
                mode = -1
                lines = []
                if mask.mean() >= 254:
                    lines = predict_line_only_img(dataset[i][0], min_line_length)
                    mode = 1
                else:
                    lines = predict_mask_img(dataset[i][0], min_line_length)
                    mode = 2
                filtered = []
                for line in lines:
                    for point in line:
                        start = int(point[0] * slice_size)
                        end = int((1 - point[1]) * slice_size)
                        radius = 3
                        radius_pixel = image_bk_gray[
                            max(end - radius, 0) : min(end + radius, slice_size - 1),
                            max(start - radius, 0) : min(start + radius, slice_size - 1),
                        ]
                        # if (row_idx, col_idx) == (4, 1):
                        #     print(line, radius_pixel, start, end)
                        if radius_pixel.size > 0 and radius_pixel.mean() == 255:
                            filtered.append(False)
                            break
                    else:
                        filtered.append(True)
                # if (row_idx, col_idx) == (4, 1):
                #     print(lines)
                #     exit()
                lines = lines[filtered]
                # if row_idx == 0 and col_idx == 0:
                if len(lines) > 0:
                    local_lines[(row_idx, col_idx)] = (lines, anchor)
                if [row_idx, col_idx] in [
                    debug_cell,
                    [debug_cell[0] + 1, debug_cell[1]],
                    [debug_cell[0], debug_cell[1] + 1],
                ]:
                    print(mode)
                    print("mask mean value:", mask.mean())
                    print("lines")
                    print(lines)
                    # print("anchor")
                    # print(lines * slice_size + anchor)
                    # print("line only")
                    # l = predict_line_only_img(dataset[i][0])
                    # plot_images(draw_line(dataset[i][0].copy(), l, thickness=1), 300)
                    # print("mask")
                    # l = predict_mask_img(dataset[i][0])
                    # plot_images(draw_line(dataset[i][0].copy(), l, thickness=1), 300)
                    plot_images(draw_line(dataset[i][0].copy(), lines, thickness=1), 300)

                image = draw_line(image, lines, thickness=2)

            image = np.concatenate((image, mask), axis=-1)
            if row_idx % interval == 0 and col_idx % interval == 0:
                image_set.append(image)

        # remove silimar lines
        threhold = 3
        for i, j in itertools.product(range(num_row), range(num_column)):
            if (i, j) not in local_lines:
                continue
            group, group_anchor = local_lines[(i, j)]
            filtered = []
            if (i, j + 1) in local_lines:
                group_right, group_right_anchor = local_lines[(i, j + 1)]
                for line in group:
                    line_g = line * slice_size + group_anchor
                    for rline in group_right:
                        rline_g = rline * slice_size + group_right_anchor
                        if (
                            norm1(line_g[0], rline_g[0]) < threhold
                            and norm1(line_g[1], rline_g[1]) < threhold
                        ) or (
                            norm1(line_g[0], rline_g[1]) < threhold
                            and norm1(line_g[1], rline_g[0]) < threhold
                        ):
                            filtered.append(False)
                            # print(line, rline_g)
                            # print(i,j)
                            # exit()
                            break
                    else:
                        filtered.append(True)
                group = group[filtered]
            filtered = []
            if (i + 1, j) in local_lines:
                group_bottom, group_bottom_anchor = local_lines[(i + 1, j)]
                for line in group:
                    line_g = line + group_anchor
                    for rline in group_bottom:
                        rline_g = rline * slice_size + group_bottom_anchor
                        if (
                            norm1(line_g[0], rline_g[0]) < threhold
                            and norm1(line_g[1], rline_g[1]) < threhold
                        ) or (
                            norm1(line_g[0], rline_g[1]) < threhold
                            and norm1(line_g[1], rline_g[0]) < threhold
                        ):
                            filtered.append(False)
                            # print(line, rline_g)
                            # print(i, j)
                            # exit()
                            break
                    else:
                        filtered.append(True)
                group = group[filtered]
            local_lines[(i, j)] = (group, local_lines[(i, j)][1])

        # combine lines between slices
        if strict_match:
            threshold = strict_match_threshold
            for i, j in itertools.product(range(num_row), range(num_column)):
                if (i, j) not in local_lines:
                    continue
                on_border_set = local_lines[(i, j)][0]
                if (i, j + 1) in local_lines:
                    matches_left = []
                    # grid right
                    for line in local_lines[(i, j + 1)][0]:
                        for a in line:
                            if a[0] <= threshold:
                                matches_left.append(a)
                    qualified = []
                    for line in on_border_set:
                        for aidx, a in enumerate(line):
                            if a[0] >= 1 - threshold:
                                qualified.append(a)
                    for line in on_border_set:
                        for aidx, a in enumerate(line):
                            if a[0] >= 1 - threshold:
                                if len(matches_left) > 0:
                                    shift_match = np.array(matches_left)
                                    shift_match[:, 0] += 1
                                    matched_distances = distance.cdist([a], shift_match)
                                    m = matches_left[np.argmin(matched_distances)]
                                    if abs(line[0, 1] - line[1, 1]) < abs(line[0, 0] - line[1, 0]):
                                        a[0] = 1
                                        a[1] = m[1]
                                        m[0] = 0
                                    else:
                                        if soft_match:
                                            shift_m = m.copy()
                                            shift_m[0] = 1
                                            dists = distance.cdist(line, [shift_m])
                                            if aidx == np.argmin(dists):
                                                local_dists = distance.cdist(qualified, [shift_m])
                                                if (qualified[np.argmin(local_dists)] == a).all():
                                                    a[0] = 1
                                                    a[1] = m[1]
                                                    m[0] = 0

                if (i + 1, j) in local_lines:
                    matches_bottom = []
                    # grid bottom
                    for line in local_lines[(i + 1, j)][0]:
                        for a in line:
                            if a[1] >= 1 - threshold:
                                matches_bottom.append(a)
                    qualified = []
                    for line in on_border_set:
                        for aidx, a in enumerate(line):
                            if a[1] <= threshold:
                                qualified.append(a)
                    for line in on_border_set:
                        for aidx, a in enumerate(line):
                            if a[1] <= threshold:
                                if len(matches_bottom) > 0:
                                    shift_match = np.array(matches_bottom)
                                    shift_match[:, 1] = 1 - shift_match[:, 1]
                                    matched_distances = distance.cdist([a], shift_match)
                                    m = matches_bottom[np.argmin(matched_distances)]
                                    if abs(line[0, 1] - line[1, 1]) > abs(line[0, 0] - line[1, 0]):
                                        a[0] = m[0]
                                        a[1] = 0
                                        m[1] = 1
                                        # if [i, j] == [2, 6]:
                                        #     print(on_border_set)
                                        #     print(matches_bottom)
                                        #     print(shift_match)
                                        #     print(matches_bottom)
                                        #     exit()
                                    else:
                                        if soft_match:
                                            shift_m = m.copy()
                                            shift_m[1] = 2
                                            dists = distance.cdist(line, [shift_m])
                                            if aidx == np.argmin(dists):
                                                local_dists = distance.cdist(qualified, [shift_m])
                                                if (qualified[np.argmin(local_dists)] == a).all():
                                                    a[0] = m[0]
                                                    a[1] = 0
                                                    m[1] = 1

                if [i, j] == debug_cell:
                    print("group")
                    print(on_border_set)
                    print("matches left")
                    print(matches_left)
                    print("matches bottom")
                    print(matches_bottom)
        local_connection_group = {}
        for pos, (lines, anchor) in list(local_lines.items()):
            connection = build_connection(
                lines,
                norm1,
                similar_threshold=0,
                threshold=local_threshold,
                duplicate_threshold=local_threshold,
            )
            local_connection_group[pos] = (connection, anchor)
            # if pos == (3, 7):
            #     print(lines)
            #     print(connection)
            #     exit()
        # convert to global coordinate
        global_line = []
        for pos, (lines, anchor) in list(local_lines.items()):
            for line in lines:
                line[0][0] = anchor[0] + line[0][0] * slice_size
                line[0][1] = anchor[1] + line[0][1] * slice_size
                line[1][0] = anchor[0] + line[1][0] * slice_size
                line[1][1] = anchor[1] + line[1][1] * slice_size
                line[:, 0] /= ori_img.shape[1]
                line[:, 1] /= ori_img.shape[0]
                global_line.append(line.tolist())
        global_connection = []
        for pos, (groups, anchor) in list(local_connection_group.items()):
            for i in range(len(groups)):
                groups[i] = np.array(groups[i])
            for group in groups:
                group[:, 0] = group[:, 0] * slice_size + anchor[0]
                group[:, 1] = group[:, 1] * slice_size + anchor[1]
                group[:, 0] /= ori_img.shape[1]
                group[:, 1] /= ori_img.shape[0]
                global_connection.append(group.tolist())
            # if pos == (2, 1) or pos == (2, 2):
            #     print(groups)
        group_connection = build_connection(
            global_connection,
            norm1,
            similar_threshold=-1,
            threshold=global_threshold,
            duplicate_threshold=global_threshold if remove_duplicate else -1,
        )
        # print(group_connection)
        # for i, group in enumerate(group_connection):
        #     color = color_map(i)
        #     ori_img = draw_rect(ori_img, group, color=color, width=5)
        plot_images(ori_img, 800)
        # line graph
        img = np.full(ori_img.shape, 255, np.uint8)
        img = draw_line(img, global_line, endpoint=True, endpoint_thickness=1)
        plot_images(img, 800)
        print("global connection map")
        img_bk = img.copy()
        for i, group in enumerate(global_connection):
            color = color_map(i % 3)
            img_bk = draw_rect(img_bk, group, color=color, width=6)
        plot_images(img_bk, 800)
        print("group connection map")
        for i, group in enumerate(group_connection):
            color = color_map(i)
            img = draw_rect(img, group, color=color, width=6)
        plot_images(img, 800)
        # exit()
        img = np.full(ori_img.shape, 255, np.uint8)
        img = draw_line(img, global_line, endpoint=True, endpoint_thickness=4, color=(0, 0, 0))

        fimg = create_grid(
            image_set,
            nrow=num_column,
            padding=1,
            pad_value=127,
        )

        plot_images(fimg, 800)
        # process_images = []
        # for i, group in enumerate(group_connection):
        #     img_part = img.copy()
        #     img_part = draw_rect(img_part, group, color=(192, 91, 22), width=5)
        #     process_images.append(img_part)
        # plot_images(process_images[0:3], img_width=800)
        # exit()
        current_shape = ori_img.shape
        for group in group_connection:
            for i, point in enumerate(group):
                x = point[0] * current_shape[1] - shift_x
                y = point[1] * current_shape[0] - shift_y
                group[i] = [x, y]
        for i, group in enumerate(group_connection):
            color = color_map(i)
            input_img = draw_rect(input_img, group, color=color, width=7, scale=False)
        plot_images(input_img, 800)
        return group_connection, input_img


def calculate_optimal_shift(slice_size, ori_img, boundary, debug):
    with HiddenPrints(disable=debug), HiddenPlots(disable=debug):
        # gray_scale = ori_img[]
        gray_scale = cv2.cvtColor(ori_img, cv2.COLOR_BGR2GRAY) / 255
        mask = ori_img[:, :, 3] < 255
        gray_scale[mask] = 1
        gray_scale = 1 - gray_scale
        plot_images(gray_scale, 600)
        # weight_metric = np.concatenate((np.linspace(0, 1, 25), np.linspace(1, 0, 25)))
        weight_metric = np.zeros(slice_size)
        weight_metric[:boundary] = -1
        weight_metric[-boundary:] = -1
        if debug:
            ax = sns.barplot(weight_metric)
            ax.set(title="weight metric", xticks=list(range(0, len(weight_metric), 10)))
            plot_images(ax, 600)
            ax.figure.clf()
        max_score = -math.inf
        start_score = 0
        max_i = 0
        record = {}
        for i in range(slice_size):
            gray_scale = shift(gray_scale, (0, 1), 0)
            sum_of_row = np.sum(gray_scale, axis=1)
            score = np.tile(weight_metric, gray_scale.shape[0] // slice_size) * sum_of_row
            record[i] = score.reshape(slice_size, -1).sum(axis=0)
            score = np.sum(score)
            if score > max_score:
                max_score = score
                max_i = i
            if i == 0:
                start_score = score
        optimal_shift_y = max_i
        # print("start_score of y:", start_score)
        # plot_images(ax := sns.barplot(record[0]), 600)
        # ax.figure.clf()
        # print("max_score of y:", max_score)
        # plot_images(ax := sns.barplot(record[max_i]), 600)
        # ax.figure.clf()
        # print(max_i)

        max_score = -math.inf
        max_i = 0
        for i in range(slice_size):
            gray_scale = shift(gray_scale, (1, 0), 0)
            sum_of_col = np.sum(gray_scale, axis=0)
            score = np.tile(weight_metric, gray_scale.shape[1] // slice_size) * sum_of_col
            score = np.sum(score)
            if score > max_score:
                max_score = score
                max_i = i
        optimal_shift_x = max_i
        if debug:
            ax = sns.barplot(sum_of_row)
            ax.set(
                xlabel="row index",
                ylabel="sum of pixel value",
                xticks=list(range(0, len(sum_of_row), 100)),
                title="sum of pixel value in each row",
            )
            plot_images(ax, 600)
            ax.figure.clf()

        if debug:
            ax = sns.barplot(sum_of_col)
            ax.set(
                xlabel="column index",
                ylabel="sum of pixel value",
                xticks=list(range(0, len(sum_of_col), 100)),
                title="sum of pixel value in each column",
            )
            plot_images(ax, 600)
            ax.figure.clf()
        if debug:
            print("original image")
            preview_dataset = TestWindowed(ori_img, 1, slice_size)
            plot_images(
                create_grid(
                    [x[0] for x in preview_dataset],
                    nrow=gray_scale.shape[1] // slice_size,
                    padding=2,
                    pad_value=127,
                ),
                img_width=700,
            )
            print("shifted image")
            preview_dataset = TestWindowed(
                shift(ori_img, (optimal_shift_x, optimal_shift_y), fill=127), 1, slice_size
            )
            plot_images(
                create_grid(
                    [x[0] for x in preview_dataset],
                    nrow=gray_scale.shape[1] // slice_size,
                    padding=2,
                    pad_value=127,
                ),
                img_width=700,
            )

        return optimal_shift_x, optimal_shift_y


if __name__ == "__main__":
    img_names = (
        "218 82 850 1807 260 50001 50038 50119 50207 50799 42852 33748 8203 7735 7578 6826 6640 5841"
    ).split()
    # img_names = ("24023 24167 24348").split()[-2:]
    # img_names = ("24023").split()
    path = "dataset_fullimg_mask/images/circuit10188.png"
    path = "dataset_fullimg_mask/images/circuit16176.png"
    path = "dataset_fullimg_mask/images/circuit" + img_names[0] + ".png"
    path = "dataset_fullimg_mask/images/circuit24348.png"
    path = "test_images/circuit50038.png"
    path = "dataset_fullimg_mask/images/circuit56081.png"
    path = "dataset_fullimg_mask/images/circuit3244.png"
    img_name = [img_names[-6]]
    group_connection, img = analyze_connection(
        cv2.imread(path, cv2.IMREAD_UNCHANGED),
        min_line_length=0.005,
        local_threshold=0.09,
        global_threshold=0.015,
        remove_duplicate=True,
        optimal_shift=True,
        boundary=3,
        strict_match=False,
        strict_match_threshold=0.01,
        debug=True,
        debug_shift_optimization=True,
        debug_cell=[-1, -1],
    )
    plot_images(img, 800)
