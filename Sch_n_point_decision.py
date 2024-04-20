
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
                if point_set.contains(end):
                    return True
                if not line_set.intersection(end).is_empty:
                    return True
                line = LineString([start, end])
                intersection = point_set.intersection(line)
                if not intersection.is_empty and intersection != start:
                    return True
                intersection = line_set.intersection(line)
                if intersection.length > 0:
                    return True
                return False

            def add_line(start, end):
                nonlocal line_set, point_set
                start, end = Point(start), Point(end)
                if start.x == end.x or start.y == end.y:
                    if corrupt(start, end):
                        return None
                    line_set = line_set.union(LineString([start, end]))
                    point_set = point_set.union(Point(end))
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
                nonlocal line_set, point_set
                start = rng.integers(0, width, 2)
                if point_set.contains(Point(start)) or line_set.contains(Point(start)):
                    exit()
                point_set = point_set.union(Point(start))
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
                target = []
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
            line_set = MultiLineString()
            point_set = MultiPoint()
            start_all = []
            start, target, middle = draw(rng, True)
            start_all.append(start)
            add_box(start, "red", .5)
            middle_point = np.array((middle + 1) / (width + 2) * 100)
            ending_point = np.array((target + 1) / (width + 2) * 100)
            fig_clean = copy.deepcopy(fig)

            for _ in range(line_num - 1):
                start, _, _ = draw(rng, False)
                add_box(start, "red", .5)
                start_all.append(start)
            start_all = np.array(start_all)
            start_all = np.array((start_all + 1) / (width + 2) * 100)
            starting_point = start_all[0]
            np.random.shuffle(start_all)
            return (plotly_to_array(fig), start_all, starting_point[0]), (plotly_to_array(fig_clean), starting_point, middle_point)

        rng = np.random.default_rng()
        while True:
            try:
                img, target = create_img_v2(rng)
                return img, target
            except StopExecution:
                pass


Datasetbehaviour.MP = True
# Datasetbehaviour.RESET = True
dataset_guise = DoubleLineFormalDataset(10000, total_output=1, line_num=2)
# plot_images([d[0][0] for d in dataset_guise[:]], -1)
# plot_images([d[1][0] for d in dataset_guise[:]], -1)
dataset_guise.view()
# %%

image_process = transforms.Compose([
    transforms.ToImage(),
    transforms.Grayscale(),
    # transforms.Resize((200, 200)),
    transforms.ToDtype(torch.float32, scale=True),
])


def xtransform(x):
    return torch.tensor(x[1]).float().cuda(), torch.tensor([x[2]]).float().cuda()


def ytransform_first(x):
    return image_process(x[0])


def ytransform_second(x):
    return torch.tensor([x[1]]).float().cuda()


def ytransform_all(x):
    return torch.tensor(x).float()


model = Model("FindMiddlePoint", dataset_guise, 256,
              xtransform=xtransform, ytransform=ytransform_second, amp=False, cudnn=True)
model.view()
# %%


class ViT_ex(nn.Module):
    def __init__(self, image_size, patch_size, dim, depth, heads, mlp_dim, result_num, dropout=0, channels=3, dim_head=64, style="decoder"):
        super().__init__()
        self.selector = nn.Sequential(
            nn.Linear(5, 2, bias=False),
            nn.ReLU(),
        )

    def forward(self, x, y):
        candidate = x[0].flatten(1)
        label = x[1]
        x = torch.cat([candidate, label], dim=1)
        x = self.selector(x)
        x = x.reshape(-1, 1, 2)
        return x


style = "decoder"
patch_size = 20
dim = int(patch_size**2 * 0.5)
m = ViT_ex(image_size=200, patch_size=patch_size, dim=dim, depth=1,
           heads=2, mlp_dim=32, channels=1, dim_head=32, dropout=0, result_num=1, style=style)
model.suffix = "_" + style
model.fit(m, nn.MSELoss(), optim.AdamW(m.parameters(), lr=1e-3),
          2000)
# %%
Datasetbehaviour.RESET = True
dataset_test = DoubleLineFormalDataset(2, total_output=1, line_num=2)
result = model.inference(dataset_test)
pprint(result[0])
