# %%
import config
from Model import *
from visualizer import get_local
from vit import Transformer


class DoubleLineFormalDataset(Datasetbehaviour):
    def __init__(self, size, total_output, line_num):
        super().__init__(size, self.__create, total_output, line_num)

    def __create(self, total_output, line_num):
        width = 20
        total_output = random.randint(1, total_output)

        def create_img_v2():
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
                def point2array(*args):
                    return LineString(
                        [
                            (args.x, args.y) if isinstance(args, Point) else np.array(args)
                            for args in args
                        ]
                    )

                nonlocal line_set, point_set
                start, end = Point(start), Point(end)
                if start.x == end.x or start.y == end.y:
                    if corrupt(start, end):
                        return None
                    line_set = line_set.union(LineString([start, end]))
                    point_set = point_set.union(Point(end))
                    fig.add_shape(
                        type="line",
                        x0=start.x,
                        y0=start.y,
                        x1=end.x,
                        y1=end.y,
                        line=dict(color="black", width=2),
                    )
                    return point2array(start, end)
                else:
                    mid = [start.x, end.y]
                    if not (corrupt(start, mid) or corrupt(mid, end)):
                        add_line(start, mid)
                        add_line(mid, end)
                        return point2array(start, mid, end)
                    mid = [end.x, start.y]
                    if not (corrupt(start, mid) or corrupt(mid, end)):
                        add_line(start, mid)
                        add_line(mid, end)
                        return point2array(start, mid, end)

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
                    fillcolor="black",
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

            def draw():
                nonlocal line_set, point_set
                lineset = []
                start = rng.integers(0, width, 2)
                if point_set.contains(Point(start)) or line_set.contains(Point(start)):
                    exit()
                point_set = point_set.union(Point(start))

                middle = rng.integers(0, width, 2)
                linepoints = add_line(start, middle)
                if linepoints is None:
                    exit()
                else:
                    lineset.append(linepoints)

                if total_output >= 4:
                    middle2 = rng.integers(0, width, 2)
                    linepoint2 = add_line(start, middle2)
                    if linepoint2 is None:
                        exit()
                    else:
                        lineset.append(linepoint2)
                # add some end point
                for _ in range(min(total_output, 3)):
                    for _ in range(5):
                        end = rng.integers(0, width, 2)
                        linepoints = add_line(middle, end)
                        if linepoints is not None:
                            lineset.append(linepoints)
                            break
                    else:
                        exit()
                for _ in range(total_output - 3):
                    for _ in range(5):
                        end = rng.integers(0, width, 2)
                        linepoints = add_line(middle2, end)
                        if linepoints is not None:
                            lineset.append(linepoints)
                            break
                    else:
                        exit()
                if total_output > 1:
                    add_point(middle)
                if total_output >= 4:
                    add_point(middle2)
                lineset = ops.linemerge(MultiLineString(lineset)).simplify(0.1)

                if isinstance(lineset, LineString):
                    return lineset
                elif isinstance(lineset, MultiLineString):
                    return list(lineset.geoms)
                else:
                    raise "Error"

            layout = go.Layout(
                xaxis=dict(
                    showline=False,
                    showgrid=False,
                    zeroline=False,
                    showticklabels=False,
                    range=[-1, width + 1],
                ),
                yaxis=dict(
                    showline=False,
                    showgrid=False,
                    zeroline=False,
                    showticklabels=False,
                    scaleanchor="x",
                    scaleratio=1,
                    range=[-1, width + 1],
                ),
                paper_bgcolor="white",
                plot_bgcolor="white",
                width=200,
                height=200,
                margin=dict(l=0, r=0, b=0, t=0, pad=0),
            )
            fig = go.Figure(layout=layout)
            line_set = MultiLineString()
            point_set = MultiPoint()

            G = nx.Graph()
            linestrings = []
            for _ in range(line_num):
                linestring = draw()
                if isinstance(linestring, list):
                    linestrings.extend(linestring)
                else:
                    linestrings.append(linestring)
                # target_all.extend(target)
                # for target_point in target:
                #     G.add_edge(start.tobytes(), target_point.tobytes())
            line_all = []
            for linestring in linestrings:
                for i in range(len(linestring.coords) - 1):
                    line_all.append([linestring.coords[i], linestring.coords[i + 1]])
                    # G.add_edge(linestring.coords[i], linestring.coords[i + 1])
            line_all = (np.array(line_all, dtype=np.float32) + 1) / (width + 2)
            target_all = []
            for line in line_all:
                G.add_edge(line[0].tobytes(), line[1].tobytes())
            for x in G.nodes:
                target_all.append(np.frombuffer(x, dtype=np.float32))
            relation = np.zeros((len(target_all), len(target_all)), dtype=bool)
            for i in range(len(target_all)):
                for j in range(len(target_all)):
                    if i == j:
                        continue
                    if G.has_edge(target_all[i].tobytes(), target_all[j].tobytes()):
                        relation[i][j] = True
                        relation[j][i] = True
            return (plotly_to_array(fig)), (np.array(target_all)), relation

        rng = np.random.default_rng()
        while True:
            try:
                img, target, meta = create_img_v2()
                return img, target, meta
            except StopExecution:
                pass


data_size = config.DATA_NUM
dataset_guise = DoubleLineFormalDataset(data_size, total_output=3, line_num=2)
result_num = 18
# %%
if config.EVAL:
    for i in range(2):
        plot_images(draw_point(dataset_guise[i][0], dataset_guise[i][1]))
        print(len(dataset_guise[i][1]))
    dataset_guise.view()
# %%

xtransform = transforms.Compose(
    [
        transforms.ToImage(),
        transforms.Grayscale(),
        transforms.ToDtype(torch.float32, scale=True),
    ]
)


def ytransform(x):
    s = np.full((result_num, 2), -1, dtype=np.float32)
    s[: len(x)] = x
    return torch.tensor(s).float()


model = Model(
    "FindPoints",
    dataset_guise,
    xtransform=xtransform,
    ytransform=ytransform,
    amp=True,
    cudnn=False,
    batch_size=config.BATCH_SIZE,
    memory_fraction=0.5,
)
model.view()
# %%
# from torchmetrics.detection import CompleteIntersectionOverUnion

# preds = [
#     {
#         "boxes": torch.tensor([[296.55, 93.96, 314.97, 152.79]]),
#     },
#     {
#         "boxes": torch.tensor([[0, 0, 1, 1]]),
#     }
# ]
# target = [
#     {
#         "boxes": torch.tensor([[300.00, 100.00, 315.00, 150.00]]),
#     },
#     {
#         "boxes": torch.tensor([[1, -1, 1, 1]]),
#     }
# ]
# # metric = CompleteIntersectionOverUnion()
# # print(metric(preds, target))
# x = torchvision.ops.complete_box_iou(preds[1]["boxes"], target[1]["boxes"])
# print(x)
# exit()

# %%


class ViT_ex(nn.Module):
    def __init__(
        self,
        image_size,
        patch_size,
        dim,
        depth,
        heads,
        dim_ff,
        result_num,
        dropout=0,
        channels=3,
        style="decoder",
    ):
        super().__init__()
        self.style = style
        image_height, image_width = image_size, image_size
        patch_height, patch_width = patch_size, patch_size

        assert (
            image_height % patch_height == 0 and image_width % patch_width == 0
        ), "Image dimensions must be divisible by the patch size."

        patch_dim = channels * patch_height * patch_width

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.to_patch_embedding = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = PositionalEncoding(dim)

        # self.transformer = Transformer(dim, depth, heads, dim_head, dim_head, dropout=dropout)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(dim, heads, dim_ff, batch_first=True, dropout=dropout),
            depth,
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(dim, heads, dim_ff, batch_first=True, dropout=dropout),
            depth,
        )
        self.decoder_query = nn.Embedding(result_num + 1, dim)

        self.box_head = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, 3)
        )
        self.relation_head = nn.Sequential(
            nn.LayerNorm(dim * 3),
            nn.Linear(dim * 3, dim),
            nn.LayerNorm(dim),
            nn.Linear(dim, 1),
        )
        # self.label_query = nn.Sequential(
        #     nn.Linear(1, dim),
        #     nn.LayerNorm(dim),
        # )

    def forward(self, x, y):
        x = self.to_patch_embedding(x)

        if self.style == "encoder":
            cls_tokens = repeat(self.cls_token, "1 1 d -> b 1 d", b=x.shape[0])
            x = torch.cat((cls_tokens, x), dim=1)
            x += self.pos_embedding(x)
            x = self.transformer(x)
            x = x[:, :1]
            x = self.mlp(x)

        elif self.style == "decoder":
            x += self.pos_embedding(x)
            x = self.transformer(x)
            tgt = repeat(self.decoder_query.weight, "d e -> n d e", n=x.shape[0])
            x = self.decoder(tgt, x)
            # x = self.mlp(x)
            # x = self.box_head(x)

        elif self.style == "decoder_only":
            # xpos = y[:, :, :1]
            # xpos = self.label_query(xpos)
            x += self.pos_embedding(x)
            # x = torch.cat((x, xpos), dim=1)

            tgt = repeat(self.decoder_query.weight, "d e -> n d e", n=x.shape[0])
            x = self.decoder(tgt, x)
        # print(y[0])
        return x


# def Hungarian_Order(g1b, g2b, criterion):
#     # cost matrix
#     T = np.zeros((len(g1b[0]), len(g1b[0])))
#     indices = []
#     for idx, (g1, g2) in enumerate(zip(torch.as_tensor(g1b), torch.as_tensor(g2b))):
#         for i, ix in enumerate(g1):
#             for j, jx in enumerate(g2):
#                 T[i][j] = criterion(ix, jx)
#         indices.append(linear_sum_assignment(T))
#     return [
#         (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
#         for i, j in indices
#     ]


def Hungarian_Order(g1b, g2b):
    indices = []
    cost_bbox = torch.cdist(g1b, g2b, p=1)
    C = cost_bbox.cpu().detach()
    indices = [linear_sum_assignment(c)[1] for c in C]
    for i in range(len(indices)):
        ind = indices[i]
        g2b[i] = g2b[i][ind]
    return indices


stage = 0


def criterion(y_hat, y, meta):
    box_head_result = meta.model.box_head(y_hat[:, :result_num])
    predicted_box, predicted_label_logit = box_head_result[:, :, :2], box_head_result[:, :, 2]
    predicted_label = torch.sigmoid(predicted_label_logit) < 0.5
    indices = Hungarian_Order(predicted_box, y)
    gtruth_label = y[:, :, 0] > 0
    gtruth_label = gtruth_label.to(torch.float32)
    predicted_box[predicted_label] = y[predicted_label].to(predicted_box.dtype)

    loss_box = F.smooth_l1_loss(predicted_box, y)
    loss_class = F.binary_cross_entropy_with_logits(predicted_label_logit, gtruth_label)
    if stage == 0:
        return 5 * loss_box + loss_class
    elif stage == 1:
        valid_relation_num = 2
        non_valid_relation_num = 4
        joint_token_list = []
        label_list = []
        for i in range(len(indices)):
            valid_relation = []
            non_valid_relation = []
            element_with_order = list(enumerate(indices[i]))
            np.random.shuffle(element_with_order)
            for c1, c2 in itertools.combinations(element_with_order, 2):
                c1n = c1[1]
                c2n = c2[1]
                box_pair = [c1[0], c2[0]]
                num_box = len(meta.data[i].meta)
                if c1n < num_box and c2n < num_box and meta.data[i].meta[c1n][c2n]:
                    valid_relation.append(box_pair)
                else:
                    non_valid_relation.append(box_pair)

            for times, relation, label in [
                (valid_relation_num, valid_relation, 1),
                (non_valid_relation_num, non_valid_relation, 0),
            ]:
                label = torch.tensor([label], dtype=torch.float32)
                for time in range(times):
                    relation_from = relation[time][0]
                    relation_to = relation[time][1]
                    obj_token_1 = y_hat[i][relation_from]
                    obj_token_2 = y_hat[i][relation_to]
                    rln_token = y_hat[i][-1]
                    joint_token = torch.cat((obj_token_1, obj_token_2, rln_token))
                    joint_token_inv = torch.cat((obj_token_2, obj_token_1, rln_token))
                    joint_token_list.append(joint_token)
                    joint_token_list.append(joint_token_inv)
                    label_list.append(label)
                    label_list.append(label)
        loss_relation = F.binary_cross_entropy_with_logits(
            meta.model.relation_head(torch.stack(joint_token_list)),
            torch.stack(label_list).cuda(),
        )
    return 5 * loss_box + loss_class + loss_relation

    for y1, y2 in zip(y_hat, y):
        predicted_box = meta["model"].box_head(y1)
        # print(predicted_box)
        print(y2)
        exit()
        total_loss = total_loss + F.mse_loss(predicted_box, y2)
        # loss = 1 - loss
        # print(predicted_box, y2)
        # exit()
        # assignment = linear_sum_assignment(loss.detach().cpu().numpy())
        # print(y1, y2)
        # print(loss)
        # print(assignment)
        # exit()
        # assignment_loss = loss[assignment]
        # total_loss += assignment_loss.mean()
    total_loss = total_loss / len(y_hat)
    return total_loss


get_local.deactivate()
style = "decoder"
# m = ViT_ex(image_size=200, patch_size=patch_size, dim=dim, depth=1,
#            heads=8, mlp_dim=32, channels=1, dim_head=32, dropout=0.1, result_num=15, style=style)
head_num = 8
dim_head = 50
m = ViT_ex(
    image_size=200,
    patch_size=10,
    dim=head_num * dim_head,
    depth=6,
    heads=head_num,
    dim_ff=head_num * dim_head,
    result_num=result_num,
    channels=1,
    dropout=0,
    style=style,
)
model.suffix = "_" + style
model.fit(
    m,
    criterion,
    optim.AdamW(m.parameters(), lr=config.LEARNING_RATE),
    2000,
    pretrained_path=config.PRETRAINED_PATH,
    keep=not config.EVAL,
    backprop_freq=config.BATCH_STEP,
)
exit()
# %%
get_local.activate()
Datasetbehaviour.RESET = True
dataset_test = DoubleLineFormalDataset(10, total_output=3, line_num=2)
result = model.inference(dataset_test)

# print(model.model.box_head(result[0][1][:4]))
# print(model.model.relation_head(torch.cat((result[0][1][0], result[0][1][1], result[0][1][-1]))))
canvas = []
for i, d in enumerate(dataset_test):
    image = dataset_test[i][0]
    image_line = image.copy()
    latent = result[i][1]
    box_latent = model.model.box_head(latent[:result_num])
    box = box_latent[:, :2]
    classes = F.sigmoid(box_latent[:, 2]) > 0.5
    box[~classes] = 0
    image = draw_point(image, box)
    # for a, b in itertools.combinations(latent[:result_num], 2):
    #     relations = model.model.relation_head(torch.cat((a, b, latent[-1])))
    #     if F.sigmoid(relations) > 0.5:
    #         p1 = model.model.box_head(a).detach().cpu()
    #         p2 = model.model.box_head(b).detach().cpu()
    #         if F.sigmoid(p1[2]) > 0.5 and F.sigmoid(p2[2]) > 0.5:
    #             p1_pos, p2_pos = p1[:2].numpy(), p2[:2].numpy()
    #             p1_pos[1] = 1 - p1_pos[1]
    #             p2_pos[1] = 1 - p2_pos[1]
    #             p1_pos *= 200
    #             p2_pos *= 200
    #             p1_pos = p1_pos.astype(int)
    #             p2_pos = p2_pos.astype(int)
    #             cv.line(image_line, p1_pos, p2_pos, color=(0, 0, 255), thickness=2)
    # attention_map = get_local.cache["Attention.forward"][0][i][0][0][:25].reshape(5, 5)
    canvas.append([image, image_line])
visualize_attentions(canvas)
# pprint(result)
