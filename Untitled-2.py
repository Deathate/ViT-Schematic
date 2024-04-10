# %%

from Model import *


class DoubleLineFormalDataset(Datasetbehaviour):
    def __init__(self, size):
        super().__init__(size, self.__create)

    def __create(self):
        width = 10
        layout = go.Layout(
            xaxis=dict(
                showline=False,
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                range=[-1, 11]),
            yaxis=dict(
                showline=False,
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                scaleanchor="x", scaleratio=1,
                range=[-1, 11]
            ),
            paper_bgcolor='white',
            plot_bgcolor='white',
            width=200,
            height=200,
            margin=dict(l=0, r=0, b=0, t=0, pad=0),
        )

        def create_img_v2(rng):
            def corrupt(start, end):
                end = Point(end)
                if p.contains(end):
                    return True
                if not l.intersection(end).is_empty:
                    return True
                line = LineString([start, end])
                intersection = p.intersection(line)
                if not intersection.is_empty and intersection != start:
                    return True
                intersection = l.intersection(line)
                if intersection.length > 0:
                    return True
                return False

            def add_line(start, end):
                nonlocal l, p
                start, end = Point(start), Point(end)
                if start.x == end.x or start.y == end.y:
                    if corrupt(start, end):
                        return None
                    l = l.union(LineString([start, end]))
                    p = p.union(Point(end))
                    fig.add_shape(type="line",
                                  x0=start.x, y0=start.y, x1=end.x, y1=end.y,
                                  line=dict(color="black", width=2)
                                  )
                    return np.array([end.x, end.y])
                else:
                    mid = [start.x, end.y]
                    if not (corrupt(start, mid) or corrupt(mid, end)):
                        add_line(start, mid)
                        add_line(mid, end)
                        return np.array(mid)
                    mid = [end.x, start.y]
                    if not (corrupt(start, mid) or corrupt(mid, end)):
                        add_line(start, mid)
                        add_line(mid, end)
                        return np.array(mid)

                return None

            def add_point(start: Annotated[list[float], 2]):
                radius = 0.15
                fig.add_shape(
                    type="circle",
                    x0=start[0] - radius,
                    y0=start[1] - radius,
                    x1=start[0] + radius,
                    y1=start[1] + radius,
                    line=dict(color="black"),  # color of the circle
                    # fill=dict(color="red")  # color of the circle
                    fillcolor="black"
                )

            def add_box(start: Annotated[list[float], 2], color, radius=0.2):
                fig.add_shape(
                    type="rect",
                    x0=start[0] - radius,
                    y0=start[1] - radius,
                    x1=start[0] + radius,
                    y1=start[1] + radius,
                    line=dict(color=color),
                    fillcolor=color,
                )

            def draw(rng, special):
                nonlocal l, p
                target = []
                start = rng.integers(0, width, 2)
                if p.contains(Point(start)) or l.contains(Point(start)):
                    exit()
                p = p.union(Point(start))
                add_box(start, "red" if special else "green", 0.5 if special else 0.2)

                for _ in range(5):
                    middle = rng.integers(0, width, 2)
                    linepoint = add_line(start, middle)
                    if linepoint is not None:
                        break
                else:
                    exit()
                # add some end point
                for _ in range(5):
                    end = rng.integers(0, width, 2)
                    linepoint = add_line(middle, end)
                    if linepoint is not None:
                        add_box(end, "gray")
                        target.append((end + 1) / (width + 2) * 100)
                        break
                else:
                    exit()
                return start, target
            fig = go.Figure(layout=layout)
            l = MultiLineString()
            p = MultiPoint()
            start, target = draw(rng, True)
            draw(rng, False)

            image_bytes = fig.to_image(format="jpg")
            image_np = np.array(Image.open(io.BytesIO(image_bytes)))
            return (image_np, np.array((start + 1) / (width + 2) * 100)), np.array(target[0])

        rng = np.random.default_rng()
        while True:
            try:
                img, target = create_img_v2(rng)
                return img, target
            except StopExecution:
                pass


d = DoubleLineFormalDataset(20000)
ipyplot.plot_images([x[0][0] for x in d[:]], img_width=100, max_images=10)

# %%
import vit


class ViT_ex(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels=3, dim_head=64):
        super().__init__()
        image_height, image_width = image_size, image_size
        patch_height, patch_width = patch_size, patch_size

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            einops.layers.torch.Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                                          p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = PositionalEncoding(dim)
        self.label_net = nn.Sequential(
            nn.Linear(2, dim),
            nn.LayerNorm(dim)
        )

        self.transformer = vit.Transformer(
            dim, depth, heads, dim_head, mlp_dim)

        self.output_net = nn.Sequential(
            nn.Linear(32, num_classes)
        )

    def forward(self, x, y):
        img, pos = x
        img = self.to_patch_embedding(img)
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=img.shape[0])
        pos = self.label_net(pos)
        pos = rearrange(pos, 'a (b c) -> a b c', b=1)
        img = torch.cat((cls_tokens, img, pos), dim=1)
        img += self.pos_embedding(img)
        img = self.transformer(img)
        img = img[:, 0]
        output = self.output_net(img)
        return output


def transform(x): return transforms.Compose([
    transforms.ToImage(),
    transforms.Grayscale(),
    transforms.ToDtype(torch.float32, scale=True),
])(x[0]).cuda(), torch.tensor(x[1]).float().cuda()


image_width = 200
patch_width = 10
m = ViT_ex(image_width, patch_width, 2, 32, 1, 1, 64, 1, 32)
model = Model("L2_P1_encoderonly", d, transform, batch_size=256, cudnn=False, amp=False)
model.fit(m, nn.MSELoss(), optim.Adam(m.parameters(), lr=0.001), 200)

# %%
get_local.activate()
size = 10
testset = DoubleLineFormalDataset(size)
inference = model.inference(testset)
images = [x[0][0] for x in testset]
result = list(zip(images, *inference))
for image, inference, truth in result:
    cv.circle(image, ((inference * 2).int().numpy()[
              0], 200 - (inference * 2).int().numpy()[1]), 5, (0, 255, 0), -1)
    cv.circle(image, [(truth * 2).int().numpy()[0],
              200 - (truth * 2).int().numpy()[1]], 8, (250, 200, 0), -1)
attention_maps = get_local.cache["Attention.forward"][0]
side_len = image_width // patch_width
maps = []
for i in tqdm(range(10)):
    map = attention_maps[i][0][0][:side_len**2].reshape(side_len, side_len)
    maps.append((images[i], map))
ipyplot.plot_images([x[0] for x in maps], img_width=120)
ipyplot.plot_images([x[1] for x in maps], img_width=120)
visualize_attentions(maps)
