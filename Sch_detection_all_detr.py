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

            return plotly_to_array(fig), np.array(target_all), G

        rng = np.random.default_rng()
        while True:
            try:
                img, target, meta = create_img_v2()
                return img, target, meta
            except StopExecution:
                pass


data_size = config.DATA_NUM
# Datasetbehaviour.RESET = True
dataset_guise = DoubleLineFormalDataset(data_size, total_output=3, line_num=2)
result_num = 18
for i in range(min(10, len(dataset_guise))):
    plot_images(draw_point(dataset_guise[i][0], dataset_guise[i][1]))
    plot_images(dataset_guise[i][0])
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


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


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
        num_classes,
        dropout=0,
        channels=3,
        style="decoder",
        box_dim=2,
    ):
        super().__init__()
        self.style = style
        image_height, image_width = image_size, image_size
        patch_height, patch_width = patch_size, patch_size

        assert (
            image_height % patch_height == 0 and image_width % patch_width == 0
        ), "Image dimensions must be divisible by the patch size."

        patch_dim = channels * patch_height * patch_width

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
        self.result_num = result_num
        self.box_dim = box_dim
        self.num_classes = num_classes
        self.box_head = MLP(dim, dim, box_dim + num_classes + 1, 3)
        self.relation_head = MLP(dim * 3, dim, 1, 3)

    def forward(self, x, y):
        x = self.to_patch_embedding(x)
        x += self.pos_embedding(x)
        x = self.transformer(x)
        tgt = repeat(self.decoder_query.weight, "d e -> n d e", n=x.shape[0])
        x = self.decoder(tgt, x)
        latent, relation_query = x[:, :-1], x[:, -1]
        outputs = self.box_head(latent)
        outputs = torch.split(outputs, [self.box_dim, self.num_classes + 1], dim=-1)
        outputs_coord, outputs_class = outputs
        out = {
            "pred_logits": outputs_class,
            "pred_boxes": outputs_coord,
            "relation_query": relation_query,
        }
        return out


class ViT_cnn(nn.Module):
    def __init__(
        self,
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
        self.to_patch_embedding = nn.Sequential(
            nn.Conv2d(channels, 4, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(4, 8, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2, 0),
            nn.ReLU(),
            Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=1, p2=1),
            nn.LayerNorm(32),
            nn.Linear(32, dim),
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
        self.decoder_query = nn.Embedding(result_num, dim)

        self.box_head = nn.Sequential(
            nn.Linear(dim, dim_ff),
            nn.ReLU(),
            nn.Linear(dim_ff, dim_ff),
            nn.ReLU(),
            nn.Linear(dim_ff, 3),
        )
        self.relation_head = nn.Sequential(
            nn.Linear(dim_ff * 3, dim_ff),
            nn.ReLU(),
            nn.Linear(dim_ff, dim_ff),
            nn.ReLU(),
            nn.Linear(dim_ff, 1),
        )

    def forward(self, x, y):
        x = self.to_patch_embedding(x)
        x += self.pos_embedding(x)
        x = self.transformer(x)
        tgt = repeat(self.decoder_query.weight, "d e -> n d e", n=x.shape[0])
        x = self.decoder(tgt, x)
        return x


from matcher import HungarianMatcher, SetCriterion

weight_dict = {"loss_ce": 1, "loss_bbox": 3}
matcher = HungarianMatcher(cost_class=0, cost_bbox=1)
criterioner = SetCriterion(1, matcher=matcher)
criterioner.to("cuda")


# def Hungarian_Order(g1b, g2b):
#     indices = []
#     cost_bbox = torch.cdist(g1b, g2b, p=1)
#     C = cost_bbox.cpu().detach()
#     indices = [linear_sum_assignment(C[i]) for i in range(len(C))]
#     return indices
#     return [
#         (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
#         for i, j in indices
#     ]
def Hungarian_Order(g1b, g2b):
    indices = []
    cost_bbox = torch.cdist(g1b, g2b, p=1)
    C = cost_bbox.cpu().detach()
    indices = [linear_sum_assignment(C[i]) for i in range(len(C))]
    for i in range(len(indices)):
        g2b[i] = g2b[i][indices[i][1]]
    return indices


def criterion(y_hat, ys, meta):
    # targets = [
    #     {
    #         "labels": torch.zeros(len(y[y[:, 0] > 0]), dtype=torch.int64).cuda(),
    #         "boxes": y[y[:, 0] > 0],
    #     }
    #     for y in ys
    # ]

    # loss_dict = criterioner(y_hat, targets)
    # losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
    # return losses

    predicted_box, predicted_label_logit = y_hat["pred_boxes"].clone(), y_hat["pred_logits"]
    predicted_bg = predicted_label_logit.softmax(-1)[:, :, 0] < 0.5
    indices = Hungarian_Order(predicted_box, ys)
    for i in range(len(indices)):
        xind, yind = indices[i]
        ys[i] = ys[i][yind]
    gtruth_label = ys[:, :, 0] > 0
    gtruth_label = gtruth_label.to(torch.long)
    predicted_box[ys[:, :, 0] < 0] = ys[ys[:, :, 0] < 0].to(predicted_box.dtype)
    predicted_box[predicted_bg] = ys[predicted_bg].to(predicted_box.dtype)
    loss_class = F.cross_entropy(predicted_label_logit.flatten(0, 1), gtruth_label.flatten())
    loss_box = F.smooth_l1_loss(predicted_box, ys, reduction="sum") / 2 / (ys[:, :, 0] > 0).sum()
    return 5 * loss_box + loss_class

    loss_box = (
        F.smooth_l1_loss(predicted_box, y, reduction="sum")
        / 2
        / (predicted_label.nelement() - predicted_label.sum())
    )
    return 3 * loss_box + loss_class
    # box_head_result = meta["model"].box_head(y_hat[:, :result_num])
    # predicted_box, predicted_label = box_head_result[:, :, :2], box_head_result[:, :, 2]
    # predicted_label_01 = F.sigmoid(predicted_label) > 0.5
    # predicted_label_01_inv = torch.logical_not(predicted_label_01)
    # loss_box_criterion = nn.SmoothL1Loss(reduction="sum")
    # Hungarian_Order(predicted_box, y, loss_box_criterion)
    # predicted_box[predicted_label_01_inv] = y[predicted_label_01_inv].to(predicted_box.dtype)
    # loss_box = loss_box_criterion(predicted_box, y)

    # if predicted_label_01.any():
    #     loss_box = loss_box / (2 * torch.count_nonzero(predicted_label_01))
    # else:
    #     loss_box = 0

    # gtruth_label = (y[:, :, 0] > 0).to(torch.float32)
    # loss_class = F.binary_cross_entropy_with_logits(predicted_label, gtruth_label)
    # return loss_box + loss_class
    # box_head_result = meta.model.box_head(y_hat[:, :result_num])
    # predicted_box, predicted_label_logit = box_head_result[:, :, :2], box_head_result[:, :, 2]
    # Hungarian_Order(predicted_box[:, :, :2], y, nn.SmoothL1Loss())
    # gtruth_label = y[:, :, 0] > 0
    # predicted_box[~gtruth_label] = y[~gtruth_label].to(predicted_box.dtype)

    # loss_class = F.binary_cross_entropy_with_logits(
    #     predicted_label_logit, gtruth_label.to(torch.float32)
    # )
    # loss_box = F.smooth_l1_loss(predicted_box, y, reduction="sum") / (gtruth_label.sum() * 2)
    # return 3 * loss_box + loss_class

    relation_loss_list = []

    valid_relation_num = 1
    non_valid_relation_num = 2
    for i, element in enumerate(y.cpu().detach().numpy()):
        valid_relation = []
        non_valid_relation = []
        element_with_order = list(enumerate(element))
        np.random.shuffle(element_with_order)
        for c1, c2 in itertools.combinations(element_with_order, 2):
            c1n = c1[1].tobytes()
            c2n = c2[1].tobytes()
            box_pair = [c1[0], c2[0]]
            if DataCell(*meta.data[i]).meta.has_edge(c1n, c2n):
                valid_relation.append(box_pair)
            else:
                non_valid_relation.append(box_pair)

        for times, relation, label in [
            (valid_relation_num, valid_relation, 1),
            (non_valid_relation_num, non_valid_relation, 0),
        ]:
            label = torch.tensor([label], dtype=torch.float32).cuda()
            for time in range(times):
                relation_from = relation[time][0]
                relation_to = relation[time][1]
                obj_token_1 = y_hat[i][relation_from]
                obj_token_2 = y_hat[i][relation_to]
                rln_token = y_hat[i][-1]
                joint_token = torch.cat((obj_token_1, obj_token_2, rln_token))
                joint_token_inv = torch.cat((obj_token_2, obj_token_1, rln_token))
                predicted_relation = meta.model.relation_head(joint_token)
                predicted_relation_inv = meta.model.relation_head(joint_token_inv)
                relation_loss_list.append(
                    F.binary_cross_entropy_with_logits(predicted_relation, label)
                )
                relation_loss_list.append(
                    F.binary_cross_entropy_with_logits(predicted_relation_inv, label)
                )

    loss_relation = torch.stack(relation_loss_list).mean()
    return 3 * loss_box + loss_class + loss_relation

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


# get_local.deactivate()
style = "decoder"
# m = ViT_ex(image_size=200, patch_size=patch_size, dim=dim, depth=1,
#            heads=8, mlp_dim=32, channels=1, dim_head=32, dropout=0.1, result_num=15, style=style)
head_num = 8
dim_head = 32
m = ViT_ex(
    image_size=200,
    patch_size=10,
    dim=head_num * dim_head,
    depth=6,
    heads=head_num,
    dim_ff=32,
    result_num=result_num,
    num_classes=1,
    channels=1,
    dropout=0.1,
    style=style,
)
# head_num = 8
# dim_head = 32
# m = ViT_cnn(
#     dim=head_num * dim_head,
#     depth=6,
#     heads=head_num,
#     dim_ff=head_num * dim_head,
#     result_num=result_num,
#     channels=1,
#     dropout=0.1,
#     style=style,
# )
model.suffix = "_" + style
pretrained_path = config.PRETRAINED_PATH
model.fit(
    m,
    criterion,
    optim.AdamW(m.parameters(), lr=config.LEARNING_RATE),
    2000,
    pretrained_path=pretrained_path,
    keep=True,
    # backprop_freq=256 / 64,
)
# %%
get_local.activate()
Datasetbehaviour.RESET = True
dataset_test = DoubleLineFormalDataset(10, total_output=1, line_num=2)
result = model.inference(dataset_test)
canvas = []
for i, d in enumerate(dataset_test):
    image = dataset_test[i][0]
    latent = result[i][1]
    box_latent = model.model.box_head(latent[:result_num])
    box = box_latent[:, :2]
    classes = F.sigmoid(box_latent[:, 2]) > 0.5
    box[~classes] = 0
    image = draw_point(image, box)
    image_line = image.copy()
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
