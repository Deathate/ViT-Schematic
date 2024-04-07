# %%
# from Model import *


# class LineDataset(Datasetbehaviour):
#     def __init__(self, size, width, L, P):
#         super().__init__(self.__create, size, width, L, P, "dataset")

#     def __create(self, size, width, L, P):
#         def create_img_v2(rng):
#             def corrupt(start, end):
#                 end = Point(end)
#                 if p.contains(end):
#                     return True
#                 if not l.intersection(end).is_empty:
#                     return True
#                 line = LineString([start, end])
#                 intersection = p.intersection(line)
#                 if not intersection.is_empty and intersection != start:
#                     return True
#                 intersection = l.intersection(line)
#                 if intersection.length > 0:
#                     return True
#                 return False

#             def add_line(start, end):
#                 nonlocal l, p
#                 start, end = Point(start), Point(end)
#                 if start.x == end.x or start.y == end.y:
#                     if corrupt(start, end):
#                         return None
#                     l = l.union(LineString([start, end]))
#                     p = p.union(Point(end))
#                     fig.add_shape(type="line",
#                                   x0=start.x, y0=start.y, x1=end.x, y1=end.y,
#                                   line=dict(color="black", width=2)
#                                   )
#                     return np.array([end.x, end.y])
#                 else:
#                     mid = [start.x, end.y]
#                     if not (corrupt(start, mid) or corrupt(mid, end)):
#                         add_line(start, mid)
#                         add_line(mid, end)
#                         return np.array(mid)
#                     mid = [end.x, start.y]
#                     if not (corrupt(start, mid) or corrupt(mid, end)):
#                         add_line(start, mid)
#                         add_line(mid, end)
#                         return np.array(mid)

#                 return None

#             def add_point(start: Annotated[list[float], 2]):
#                 radius = 0.2
#                 fig.add_shape(
#                     type="circle",
#                     x0=start[0] - radius,
#                     y0=start[1] - radius,
#                     x1=start[0] + radius,
#                     y1=start[1] + radius,
#                     line=dict(color="black"),  # color of the circle
#                     # fill=dict(color="red")  # color of the circle
#                     fillcolor="black"
#                 )

#             def add_box(start: Annotated[list[float], 2], color, radius=0.2):
#                 fig.add_shape(
#                     type="rect",
#                     x0=start[0] - radius,
#                     y0=start[1] - radius,
#                     x1=start[0] + radius,
#                     y1=start[1] + radius,
#                     line=dict(color=color),
#                     fillcolor=color,
#                 )

#             def draw(rng, special):
#                 nonlocal l, p
#                 target = []
#                 start = rng.integers(0, width, 2)
#                 if p.contains(Point(start)) or l.contains(Point(start)):
#                     exit()
#                 p = p.union(Point(start))
#                 # add_box(start, "red" if special else "green", 0.5 if special else 0.2)

#                 endpoint_num = rng.integers(2, max(3, P + 1))
#                 if not special:
#                     endpoint_num = 5

#                 middle = rng.integers(0, width, 2)
#                 linepoint = add_line(start, middle)

#                 if linepoint is None:
#                     exit()
#                 if endpoint_num >= 4:
#                     middle2 = rng.integers(0, width, 2)
#                     linepoint2 = add_line(start, middle2)
#                     if linepoint2 is None:
#                         exit()
#                 # add some end point
#                 for _ in range(min(endpoint_num, 3)):
#                     for _ in range(5):
#                         end = rng.integers(0, width, 2)
#                         linepoint = add_line(middle, end)
#                         if linepoint is not None:
#                             add_box(end, "gray")
#                             target.append((end + 1) / (width + 2) * 100)
#                             break
#                     else:
#                         exit()
#                 if endpoint_num > 1:
#                     add_point(middle)
#                 for _ in range(max(endpoint_num - 3, 0)):
#                     for _ in range(5):
#                         end = rng.integers(0, width, 2)
#                         linepoint = add_line(middle2, end)
#                         if linepoint is not None:
#                             add_box(end, "gray")
#                             target.append((end + 1) / (width + 2) * 100)
#                             break
#                     else:
#                         exit()
#                 if endpoint_num > 4:
#                     add_point(middle2)
#                 return start, target
#             layout = go.Layout(
#                 xaxis=dict(
#                     showline=False,
#                     showgrid=False,
#                     zeroline=False,
#                     showticklabels=False,
#                     range=[-1, width + 1]),
#                 yaxis=dict(
#                     showline=False,
#                     showgrid=False,
#                     zeroline=False,
#                     showticklabels=False,
#                     scaleanchor="x", scaleratio=1,
#                     range=[-1, width + 1]
#                 ),
#                 paper_bgcolor='white',
#                 plot_bgcolor='white',
#                 width=200,
#                 height=200,
#                 margin=dict(l=0, r=0, b=0, t=0, pad=0),
#             )
#             fig = go.Figure(layout=layout)
#             l = MultiLineString()
#             p = MultiPoint()
#             start, target = draw(rng, True)
#             start = np.array((start + 1) / (width + 2) * 100)
#             for _ in range(L):
#                 draw(rng, False)

#             image_bytes = fig.to_image(format="jpg")
#             image_np = np.array(Image.open(io.BytesIO(image_bytes)))
#             target_padding = np.zeros((6, 2))
#             target_padding.fill(-1)
#             target = np.array(target)
#             target_padding[:len(target)] = target
#             return (image_np, start, target_padding), target_padding

#         def main(_):
#             rng = np.random.default_rng()
#             while True:
#                 try:
#                     img, target = create_img_v2(rng)
#                     return img, target
#                 except StopExecution:
#                     pass
#         self.dataset = p_tqdm.p_umap(main, range(size))


# d = LineDataset(20000, 25, 2, 6)
# d = LineDataset(10, 25, 2, 6)
# d = LineDataset(20, 25, 2, 6)
# d = LineDataset(30, 25, 2, 6)
# ipyplot.plot_images([x[0][0] for x in d], img_width=200, labels=[
#                     str((x[1])) for x in d], max_images=10)

# %%
from Model import *


class MessyDataset(Datasetbehaviour):
    def __init__(self, size):
        self.postive = [
            cv.imread(str(x), cv.IMREAD_UNCHANGED)
            for x in list(Path("data_cleaning_example/positive").glob("*/*.png"))
        ]
        self.negative = [
            cv.imread(str(x), cv.IMREAD_UNCHANGED)
            for x in list(Path("data_cleaning_example/negative").glob("*/*png"))
        ]
        super().__init__(size, self.__create)

    def __create(self):
        def add_line(img, start_point, end_point):
            # Define color and thickness of the line
            color = (0, 0, 0, 255)
            thickness = 1
            cv2.line(
                img,
                np.int32(start_point),
                np.int32(end_point),
                color,
                thickness,
                lineType=cv.LINE_AA,
            )

        def add_dottedline(img, start_point, end_point):
            # Define color and thickness of the line
            color = (0, 0, 0, 255)
            thickness = 2

            # Define length of the dotted segments and space between them
            dot_length = 1
            dot_space = 10

            # Calculate the length of the line
            line_length = (
                (end_point[0] - start_point[0]) ** 2 + (end_point[1] - start_point[1]) ** 2
            ) ** 0.5

            # Calculate the number of segments
            num_segments = int(line_length / (dot_length + dot_space))

            # Calculate the x and y increments
            x_increment = (end_point[0] - start_point[0]) / num_segments
            y_increment = (end_point[1] - start_point[1]) / num_segments

            # Draw the dotted line
            for i in range(num_segments):
                start_dot = (
                    int(start_point[0] + i * x_increment),
                    int(start_point[1] + i * y_increment),
                )
                end_dot = (
                    int(start_point[0] + (i + 0.5) * x_increment),
                    int(start_point[1] + (i + 0.5) * y_increment),
                )
                cv2.line(img, start_dot, end_dot, color, thickness)

        def add_box(img, start_point, end_point):
            # box = shapely.box(*start_point, *end_point)
            # discriminator = shapely.union(discriminator, box)
            # Define color and thickness of the rectangle
            color = (0, 0, 0, 255)  # Green color in BGR
            thickness = 2
            # Draw the rectangle
            cv2.rectangle(img, start_point, end_point, color, thickness)

        def add_circle(img, start_point, radius):
            color = (0, 0, 0, 255)
            thickness = 2
            cv2.circle(img, start_point, radius, color, thickness)

        def create_endpoints():
            while True:
                start = rng.integers(30, 480, size=2)
                end = rng.integers(30, 480, size=2)
                if abs(end[0] - start[0]) < 20 or abs(end[1] - start[1]) < 20:
                    continue
                return start, end

        def affine_image(image, angle, scale=1.0):
            from skimage.transform import rotate

            def reshape_to_square(image, desired_size):
                # old_size is in (height, width) format
                old_size = image.shape[:2]

                ratio = float(desired_size) / max(old_size)
                new_size = tuple([int(x * ratio) for x in old_size])

                # Resize image while preserving aspect ratio
                image = cv2.resize(image, (new_size[1], new_size[0]))

                desired_size = int(desired_size)
                delta_w = desired_size - new_size[1]
                delta_h = desired_size - new_size[0]
                top, bottom = delta_h // 2, delta_h - (delta_h // 2)
                left, right = delta_w // 2, delta_w - (delta_w // 2)

                # Pad the image to make it square
                color = [0, 0, 0]
                squared_image = cv2.copyMakeBorder(
                    image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
                )

                return squared_image

            h, w = np.array(image.shape[:2])
            image = reshape_to_square(image, max(h, w))
            if scale <= 100:
                scaled_width = np.int32((image.shape[0] * scale, image.shape[1] * scale))
                image = cv.resize(image, scaled_width)
            else:
                image = cv.resize(image, (scale, scale))
            image = rotate(image, angle, resize=True) * 255
            return image

        def add_image(canvas, image, start):
            h, w = image.shape[:2]
            for i in range(start[0], start[0] + h):
                for j in range(start[1], start[1] + w):
                    if image[i - start[0], j - start[1], 3] > 0:
                        canvas[i, j] = image[i - start[0], j - start[1]]

        def add_random_image(canvas, candicates, scale_range=[0.99, 1]):
            angle = random.choice([0, 90, 180, 270])
            cand = random.choice(candicates)
            scale = rng.uniform(*scale_range)
            image = affine_image(cand, angle, min(scale, 480 / cand.shape[0]))
            start = rng.integers(512 - max(*image.shape), size=2)
            add_image(canvas, image, start)
        rng = np.random.default_rng()
        img = np.full((512, 512, 4), 255, np.uint8)

        # for _ in range(rng.integers(3)):
        #     start, end = create_endpoints()
        #     radius = rng.integers(5, 40)
        #     add_circle(img, start, radius)

        # for _ in range(rng.integers(5)):
        #     start, end = create_endpoints()
        #     add_box(img, start, end)
        for _ in range(2):
            add_random_image(img, self.postive, [0.8, 1.2])

        img2 = np.copy(img)
        for _ in range(2):
            start, end = create_endpoints()
            add_line(img, start, end)
        for _ in range(2):
            start, end = create_endpoints()
            add_dottedline(img, start, end)
        for _ in range(20):
            add_random_image(img, self.negative, [0.8, 1.2])
        img = cv.cvtColor(img, cv.COLOR_RGBA2RGB)
        img2 = cv.cvtColor(img2, cv.COLOR_RGBA2RGB)
        return img, img2


class GroundTruthDataset(Datasetbehaviour):
    def __init__(self, size):
        self.i = 0
        self.ground = [
            cv.imread(str(x), cv.IMREAD_UNCHANGED)
            for x in list(x for x in Path("data_cleaning_example/ground_truth").glob('*jpg') if 'c' not in x.name)
        ]
        self.truth = [
            cv.imread(str(x), cv.IMREAD_UNCHANGED)
            for x in list(Path("data_cleaning_example/ground_truth").glob('*c*'))
        ]
        size = len(self.ground)
        super().__init__(size, self.__create)

    def __create(self):
        res = self.ground[self.i], self.truth[self.i]
        self.i += 1
        return res


class TestDataset(Datasetbehaviour):
    def __init__(self, size):
        self.i = 0
        self.library = [
            "data_cleaning_example/dac082s085-page29_SOIC_Section_0.png",
            "data_cleaning_example/dac082s085-page29_SOIC_Short_0.png",
            "data_cleaning_example/dac082s085-page29_SOIC_Top_0.png",
        ]
        size = len(self.library)
        super().__init__(size, self.__create)

    def __create(self):
        img = cv.imread(self.library[self.i])
        res = img, img
        self.i += 1
        return res


if __name__ == "__main__":
    plot_images(MessyDataset(10), 200)
    plot_images(GroundTruthDataset(10), 200)
    # plot_images(TestDataset(3), 150)
