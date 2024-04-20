# %%
from Model import *


class MessyDataset(Datasetbehaviour):
    def __init__(self, size):
        self.postive = [
            cv2.imread(str(x), cv2.IMREAD_UNCHANGED)
            for x in list(Path("data_cleaning_example/positive").glob("*/*.png"))
        ]
        self.negative = [
            cv2.imread(str(x), cv2.IMREAD_UNCHANGED)
            for x in list(Path("data_cleaning_example/negative").glob("*/*png"))
        ]
        super().__init__(size, self.__create)

    def __create(self):
        rng = np.random.default_rng()

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
                lineType=cv2.LINE_AA,
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
                image = cv2.resize(image, scaled_width)
            else:
                image = cv2.resize(image, (scale, scale))
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

        img = np.full((512, 512, 4), 255, np.uint8)

        # for _ in range(rng.integers(3)):
        #     start, end = create_endpoints()
        #     radius = rng.integers(5, 40)
        #     add_circle(img, start, radius)

        # for _ in range(rng.integers(5)):
        #     start, end = create_endpoints()
        #     add_box(img, start, end)
        add_random_image(img, self.postive, [0.8, 1.2])

        img2 = np.copy(img)
        for _ in range(2):
            start, end = create_endpoints()
            add_line(img, start, end)
        for _ in range(2):
            start, end = create_endpoints()
            add_dottedline(img, start, end)
        for _ in range(10):
            add_random_image(img, self.negative, [0.8, 1.2])
        img = cv.resize(img, (200, 200))
        img2 = cv.resize(img2, (200, 200))
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_RGBA2RGB)
        return img, img2


if __name__ == "__main__":
    Datasetbehaviour.MP = True
    # Datasetbehaviour.RESET = True
    ds = MessyDataset(35000)
    plot_images(ds)
