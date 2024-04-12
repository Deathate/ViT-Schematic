
# %%
# %%
from Model import *
from visualizer import get_local
from vit import Transformer


class DoubleLineFormalDataset(Datasetbehaviour):
    def __init__(self, size, total_output, line_num):
        super().__init__(size, self.__create, total_output, line_num)

    def __create(self, total_output, line_num):
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
                add_box(start, "grey")

                endpoint_num = total_output
                if not special:
                    endpoint_num = 1

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
                            target.append(end)
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
                            target.append(end)
                            break
                    else:
                        exit()
                if endpoint_num > 4:
                    add_point(middle2)
                target = np.array(target)
                middle = np.array(middle)
                return start, target, middle
            target_all = []
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
            start, target, middle = draw(rng, True)

            add_box(start, "red", .5)
            starting_point = np.array((start + 1) / (width + 2) * 100)
            middle_point = np.array((middle + 1) / (width + 2) * 100)
            ending_point = np.array((target + 1) / (width + 2) * 100)

            for t in target:
                target_all.append(t)

            target_padding = np.full((total_output, 2), -1, dtype=np.float32)
            target_padding[:len(target)] = np.array(target)

            for _ in range(line_num - 1):
                _, target, _ = draw(rng, False)
                # add_box(start, "red", .5)
                target_all.append(start)
                for t in target:
                    target_all.append(t)

            image_bytes = fig.to_image(format="jpg")
            image_np = np.array(Image.open(io.BytesIO(image_bytes)))
            target_all = np.array(target_all)

            return image_np, (starting_point, middle_point)

        rng = np.random.default_rng()
        while True:
            try:
                img, target = create_img_v2(rng)
                return img, target
            except StopExecution:
                pass


Datasetbehaviour.reset()
Datasetbehaviour.MP = True
dataset_guise = DoubleLineFormalDataset(40000, total_output=1, line_num=2)
plot_images([d[0] for d in dataset_guise[:]], -1)
dataset_guise.view()
# %%

image_process = transforms.Compose([
    transforms.ToImage(),
    transforms.Grayscale(),
    # transforms.Resize((200, 200)),
    transforms.ToDtype(torch.float32, scale=True),
])


def transform(x):
    return image_process(x)


def ytransform(x):
    return torch.tensor(x[0]).float()


model = Model("FindMiddlePoint", dataset_guise, 256,
              xtransform=transform, ytransform=ytransform, amp=False, cudnn=True)
model.view()
# %%


class ViT_ex(nn.Module):
    def __init__(self, image_size, patch_size, dim, depth, heads, mlp_dim, result_num, dropout=0, channels=3, dim_head=64, style="decoder"):
        super().__init__()
        self.style = style
        image_height, image_width = image_size, image_size
        patch_height, patch_width = patch_size, patch_size

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        patch_dim = channels * patch_height * patch_width

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                      p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = PositionalEncoding(dim)

        self.transformer = Transformer(
            dim, depth, heads, dim_head, mlp_dim, dropout=dropout)
        self.decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(
            dim, heads, dim_head, batch_first=True, dropout=dropout), depth)
        self.decoder_query = nn.Embedding(result_num, dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, result_num * 2),
            nn.Sigmoid()
        )

    def forward(self, x, y):
        x = self.to_patch_embedding(x)

        if self.style == "encoder":
            cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=x.shape[0])
            x = torch.cat((cls_tokens, x), dim=1)
            x += self.pos_embedding(x)
            x = self.transformer(x)
            x = x[:, 0]
            x = self.mlp(x)
        elif self.style == "decoder":
            x += self.pos_embedding(x)
            x = self.transformer(x)
            tgt = repeat(self.decoder_query.weight, 'd e -> n d e', n=x.shape[0])
            x = self.decoder(tgt, x)
            x = self.mlp(x)
            x = x.reshape(-1, 2)
        elif self.style == "decoder_only":
            x += self.pos_embedding(x)
            tgt = repeat(self.decoder_query.weight, 'd e -> n d e', n=x.shape[0])
            x = self.decoder(tgt, x)
            x = self.mlp(x)
            x = x.reshape(-1, 2)
        return x * 100


get_local.deactivate()
style = "decoder"
m = ViT_ex(image_size=200, patch_size=10, dim=32, depth=2,
           heads=4, mlp_dim=32, channels=1, dim_head=32, dropout=0, result_num=1, style=style)

model.suffix = "_" + style
model.fit(m, nn.MSELoss(), optim.AdamW(m.parameters(), lr=1e-3),
          2000)
# %%
get_local.activate()
Datasetbehaviour.reset()
Datasetbehaviour.RESET = True
dataset_test = DoubleLineFormalDataset(1, total_output=1, line_num=2)
dataset_test.view()
plot_images([d[0] for d in dataset_test[:]], -1)
result = model.inference(dataset_test, verbose=True)
pprint(result)
image = dataset_test[0][0]
point = result[1][0] * 2
point[1] = 200 - point[1]
pad = 7
cv.rectangle(image, (int(point[0]) - pad, int(point[1]) - pad),
             (int(point[0]) + pad, int(point[1]) + pad), (0, 255, 0), 2)
plot_images([image], -1)

# pprint(result)

# plot_images(get_local.cache["Attention.forward"][0][0][0][0].reshape(20, 20) * 1000)
# print(get_local.cache["Attention.forward"][0][0][0][0][:400].reshape(20, 20))
# print(model.inference(model[:1]))
# %%
# test_data = SchematicleGroundTruth(400, 10)
# result = model.inference(test_data[:1])
# test_data = SchematicleLineDataset(10000, line_num=2, endpoint_num=6,
#                                    width=40, image_width=400, padding=2)
# result = model.inference(test_data[:])
# a, b, c = result
# print(b)
# print(c)
# plot_images(result[])

# model.inference(dataset_guise[0], True)
# model.preprocessing(test_data[:1], False)
