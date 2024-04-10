
# %%
from createdata import SchematicleGroundTruth, SchematicleLineDataset
from Model import *
from visualizer import get_local
from vit import Transformer


class DoubleLineFormalDataset(Datasetbehaviour):
    def __init__(self, size):
        super().__init__(size, self.__create)

    def __create(self):
        width = 25

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
                radius = 0.2
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
                # add_box(start, "red" if special else "green", 0.5 if special else 0.2)

                endpoint_num = rng.integers(2, 7)
                if not special:
                    endpoint_num = 5

                middle = rng.integers(0, width, 2)
                linepoint = add_line(start, middle)

                if linepoint is None:
                    exit()
                if endpoint_num >= 4:
                    middle2 = rng.integers(0, width, 2)
                    linepoint2 = add_line(start, middle2)
                    if linepoint2 is None:
                        exit()
                # add some end point
                for _ in range(min(endpoint_num, 3)):
                    for _ in range(5):
                        end = rng.integers(0, width, 2)
                        linepoint = add_line(middle, end)
                        if linepoint is not None:
                            add_box(end, "gray")
                            target.append((end + 1) / (width + 2) * 100)
                            break
                    else:
                        exit()
                if endpoint_num > 1:
                    add_point(middle)
                for _ in range(max(endpoint_num - 3, 0)):
                    for _ in range(5):
                        end = rng.integers(0, width, 2)
                        linepoint = add_line(middle2, end)
                        if linepoint is not None:
                            add_box(end, "gray")
                            target.append((end + 1) / (width + 2) * 100)
                            break
                    else:
                        exit()
                if endpoint_num > 4:
                    add_point(middle2)
                return start, target
            layout = go.Layout(
                xaxis=dict(
                    showline=False,
                    showgrid=False,
                    zeroline=False,
                    showticklabels=False,
                    range=[-1, width + 1]),
                yaxis=dict(
                    showline=False,
                    showgrid=False,
                    zeroline=False,
                    showticklabels=False,
                    scaleanchor="x", scaleratio=1,
                    range=[-1, width + 1]
                ),
                paper_bgcolor='white',
                plot_bgcolor='white',
                width=200,
                height=200,
                margin=dict(l=0, r=0, b=0, t=0, pad=0),
            )
            fig = go.Figure(layout=layout)
            l = MultiLineString()
            p = MultiPoint()
            start, target = draw(rng, True)
            start = np.array((start + 1) / (width + 2) * 100)
            for _ in range(2):
                draw(rng, False)

            image_bytes = fig.to_image(format="jpg")
            image_np = np.array(Image.open(io.BytesIO(image_bytes)))
            target_padding = np.full((6, 2), -1)
            target_padding[:len(target)] = np.array(target)
            return (image_np, start), target_padding

        rng = np.random.default_rng()
        while True:
            try:
                img, target = create_img_v2(rng)
                return img, target
            except StopExecution:
                pass


Datasetbehaviour.MP = True
d = DoubleLineFormalDataset(10000)
ipyplot.plot_images([x[0][0] for x in d], img_width=200, labels=[
    str((x[1])) for x in d], max_images=10)

# %%
Datasetbehaviour.MP = True
dataset_guise = SchematicleLineDataset(10000, line_num=3, endpoint_num=10,
                                       width=40, image_width=400, padding=20)
plot_images([d[0][0] for d in dataset_guise[:]])
# %%


def transform(x):
    return transforms.Compose([
        transforms.ToImage(),
        transforms.Grayscale(),
        transforms.ToDtype(torch.float32, scale=True),
    ])(x[0]).cuda(), torch.tensor(x[1]).float().cuda()


model = Model("L2-5_P2-6_nomask_nov", d,
              transform, batch_size=256, amp=False, cudnn=False)


class ViT_ex(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, dropout=0, channels=3, dim_head=64):
        super().__init__()
        image_height, image_width = image_size, image_size
        patch_height, patch_width = patch_size, patch_size

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        self.num_classes = num_classes
        # self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
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

        self.transformer = Transformer(
            dim, depth, heads, dim_head, mlp_dim, dropout=dropout)
        self.decoder = nn.TransformerDecoderLayer(
            dim, heads, dim_head, batch_first=True, dropout=dropout)
        self.tgt = nn.Parameter(torch.randn(1, num_classes, dim))
        self.tgt_embedding = nn.Linear(1, dim)
        self.output_net = nn.Sequential(
            nn.Linear(dim * num_classes, num_classes),
            # nn.ReLU(),
        )

    def forward(self, x, y):
        img, pos = x
        img = self.to_patch_embedding(img)
        pos = self.label_net(pos)
        pos = rearrange(pos, 'a (b c) -> a b c', b=1)
        img += self.pos_embedding(img)
        img = torch.cat((img, pos), dim=1)
        img = self.transformer(img)
        # tgt = repeat(self.tgt, '1 a b -> c a b', c=img.shape[0])
        tgt = y.detach().clone()
        tgt = einops.rearrange(tgt, 'a b c -> a (b c) 1')
        tgt = self.tgt_embedding(tgt)
        output = self.decoder(tgt, img)
        output = rearrange(output, 'a b c -> a (b c)')
        output = self.output_net(output)
        output = rearrange(output, "a (b c) -> a b c", c=2)
        # exit()
        # print(output)
        return output


get_local.deactivate()
m = ViT_ex(image_size=200, patch_size=10, num_classes=12, dim=32, depth=1,
           heads=1, mlp_dim=32, channels=1, dim_head=32, dropout=0)


model.fit(m, nn.MSELoss(), optim.Adam(m.parameters(), lr=1e-3),
          500, target_transform=Hungarian_Order)
# %%
print(model.inference(model[:1]))
# %%
test_data = SchematicleGroundTruth(400, 10)
result = model.inference(test_data[:1])
test_data = SchematicleLineDataset(1, line_num=3, endpoint_num=10,
                                   width=40, image_width=400, padding=20)
result = model.inference(test_data[:1])
print(result)
# plot_images(result[])

# model.inference(dataset_guise[0], True)
# model.preprocessing(test_data[:1], False)
