
# %%
import config
from Model import *
from visualizer import get_local
from vit import Transformer

TEST = config.TEST


class DoubleLineFormalDataset(Datasetbehaviour):
    def __init__(self, size, total_output, line_num):
        super().__init__(size, self.__create, total_output, line_num)

    def __create(self, total_output, line_num):
        width = 15

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
                def point2array(*args):
                    return LineString([(args.x, args.y) if isinstance(args, Point) else np.array(args)for args in args])
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

            def draw(rng):
                nonlocal line_set, point_set
                lineset = []
                start = rng.integers(0, width, 2)
                if point_set.contains(Point(start)) or line_set.contains(Point(start)):
                    exit()
                point_set = point_set.union(Point(start))
                # add_box(start, "grey")

                middle = rng.integers(0, width, 2)
                linepoints = add_line(start, middle)
                if linepoints is None:
                    exit()
                lineset.append(linepoints)
                # if endpoint_num >= 4:
                #     middle2 = rng.integers(0, width, 2)
                #     linepoint2 = add_line(start, middle2)
                #     if linepoint2 is None:
                #         exit()
                # add some end point
                for _ in range(min(total_output, 3)):
                    for _ in range(5):
                        end = rng.integers(0, width, 2)
                        linepoints = add_line(middle, end)
                        if linepoints is not None:
                            # add_box(end, "gray")
                            lineset.append(linepoints)
                            break
                    else:
                        exit()
                if total_output > 1:
                    add_point(middle)

                lineset = ops.linemerge(MultiLineString(lineset)).simplify(0.1)
                lineset = np.array(lineset.coords)

                # for _ in range(max(endpoint_num - 3, 0)):
                #     for _ in range(5):
                #         end = rng.integers(0, width, 2)
                #         linepoint = add_line(middle2, end)
                #         if linepoint is not None:
                #             add_box(end, "gray")
                #             target.append(end)
                #             break
                #     else:
                #         exit()
                # if endpoint_num > 4:
                #     add_point(middle2)

                # start = (np.array(start, dtype=np.float32) + 1) / (width + 2)
                lineset = (np.array(lineset, dtype=np.float32) + 1) / (width + 2)
                # middle = (np.array(middle, dtype=np.float32) + 1) / (width + 2)
                return lineset

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
            target_all = []
            # G = nx.Graph()
            for _ in range(line_num):
                linepoints = draw(rng)
                target_all.extend(linepoints)
                # add_box(start, "red", .5)
                # target_all.extend(target)
                # for target_point in target:
                #     G.add_edge(start.tobytes(), target_point.tobytes())

            np.random.shuffle(target_all)
            # target_box_all = []
            # pad = 0.03
            # for target in target_all:
            #     target_box_all.append([target[0] - pad, target[1] - pad,
            #                           target[0] + pad, target[1] + pad])
            # target_box_all = np.array(target_box_all)

            return (plotly_to_array(fig)), (np.array(target_all))

        rng = np.random.default_rng()
        while True:
            try:
                img, target = create_img_v2(rng)
                return img, target
            except StopExecution:
                pass


Datasetbehaviour.MP = True
# Datasetbehaviour.RESET = True
dataset_guise = DoubleLineFormalDataset(
    config.DATA_NUM if not TEST else 2, total_output=1, line_num=2)
# dataset_guise[0]
plot_images(draw_point(dataset_guise[0][0], dataset_guise[0][1]))
plot_images(dataset_guise[0][0])
print(dataset_guise[0][1])
# dataset_guise.view()
# %%

xtransform = transforms.Compose([
    transforms.ToImage(),
    transforms.Grayscale(),
    transforms.ToDtype(torch.float32, scale=True),
])


def ytransform(x):
    s = np.full((15, 2), -1, dtype=np.float32)
    s[:len(x)] = x
    return torch.tensor(s).float()


model = Model("FindPoints", dataset_guise, config.BATCH_SIZE,
              xtransform=xtransform, ytransform=ytransform, amp=True, cudnn=False)
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
result_num = 15

# torch.autograd.set_detect_anomaly(True)


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
        self.decoder_query = nn.Embedding(result_num + 1, dim)

        self.box_head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 3),
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
            cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=x.shape[0])
            x = torch.cat((cls_tokens, x), dim=1)
            x += self.pos_embedding(x)
            x = self.transformer(x)
            x = x[:, :1]
            x = self.mlp(x)

        elif self.style == "decoder":
            x += self.pos_embedding(x)
            x = self.transformer(x)
            tgt = repeat(self.decoder_query.weight, 'd e -> n d e', n=x.shape[0])
            x = self.decoder(tgt, x)
            # x = self.mlp(x)
            # x = self.box_head(x)

        elif self.style == "decoder_only":
            # xpos = y[:, :, :1]
            # xpos = self.label_query(xpos)
            x += self.pos_embedding(x)
            # x = torch.cat((x, xpos), dim=1)

            tgt = repeat(self.decoder_query.weight, 'd e -> n d e', n=x.shape[0])
            x = self.decoder(tgt, x)
        # print(y[0])
        return x

    # def postprocess(self, x):
    #     box_head_result = self.box_head(x)
    #     predicted_box, predicted_label = box_head_result[:, :, :2], box_head_result[:, :, 2]
    #     predicted_box = F.sigmoid(predicted_box)
    #     return predicted_box, predicted_label


def criterion(y_hat, y, meta):
    box_head_result = meta["model"].box_head(y_hat[:, :result_num])
    predicted_box, predicted_label = box_head_result[:, :, :2], box_head_result[:, :, 2]
    predicted_label_01 = (F.sigmoid(predicted_label) > 0.5)
    predicted_label_01_inv = torch.logical_not(predicted_label_01)
    loss_box_criterion = nn.SmoothL1Loss(reduction="sum")
    Hungarian_Order(predicted_box, y, loss_box_criterion)
    predicted_box[predicted_label_01_inv] = y[predicted_label_01_inv].to(predicted_box.dtype)
    loss_box = loss_box_criterion(predicted_box, y)

    if predicted_label_01.any():
        loss_box = loss_box / (2 * torch.count_nonzero(predicted_label_01))
    else:
        loss_box = 0

    gtruth_label = (y[:, :, 0] > 0).to(torch.float32)
    loss_class = F.binary_cross_entropy_with_logits(predicted_label, gtruth_label)
    return loss_box + loss_class

    # return loss_box
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
            if meta["data"][i].meta.has_edge(c1n, c2n):
                valid_relation.append(box_pair)
            else:
                non_valid_relation.append(box_pair)

        for times, relation, label in ((valid_relation_num, valid_relation, 1), (non_valid_relation_num, non_valid_relation, 0)):
            for time in range(times):
                relation_from = relation[time][0]
                relation_to = relation[time][1]
                obj_token_1 = y_hat[i][relation_from]
                obj_token_2 = y_hat[i][relation_to]
                rln_token = y_hat[i][-1]
                joint_token = torch.cat((obj_token_1, obj_token_2, rln_token))
                predicted_relation = meta["model"].relation_head(
                    joint_token)
                relation_loss_list.append(nn.BCEWithLogitsLoss()(predicted_relation,
                                                                 torch.tensor([label], dtype=torch.float32).cuda()))

    loss_relation = torch.stack(relation_loss_list).mean()
    return loss_box + loss_relation / (valid_relation_num + non_valid_relation_num)

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
patch_size = 10
dim = 256
m_detr = ViT_ex(image_size=200, patch_size=patch_size, dim=dim, depth=6,
                heads=8, mlp_dim=32, channels=1, dim_head=32, dropout=0.1, result_num=15, style=style)
m = m_detr
# m = ViT_ex(image_size=200, patch_size=patch_size, dim=64, depth=2,
#            heads=2, mlp_dim=32, channels=1, dim_head=32, dropout=0.1, result_num=15, style=style)
model.suffix = "_" + style
pretrained_path = "runs/FindPoints/0506_00-52-08__decoder/epoch=270.pth"
model.fit(m, criterion, optim.AdamW(m.parameters(), lr=config.LEARNING_RATE),
          2000, pretrained_path=pretrained_path if TEST else "")
model.view()
# %%

get_local.activate()
Datasetbehaviour.RESET = True
dataset_test = DoubleLineFormalDataset(5, total_output=1, line_num=2)
result = model.inference(dataset_test)

# print(model.model.box_head(result[0][1][:4]))
# print(model.model.relation_head(torch.cat((result[0][1][0], result[0][1][1], result[0][1][-1]))))
canvas = []
for i, d in enumerate(dataset_test):
    image = dataset_test[i][0]
    latent = result[i][1]
    box_latent = model.model.box_head(latent[:result_num])
    box = box_latent[:, :2]
    box = torch.sigmoid(box)
    classes = F.sigmoid(box_latent[:, 2]) > 0.5
    for j in range(len(classes)):
        if not classes[j]:
            box[j] = 0

    Hungarian_Order(box, result[i][2])
    print(box)
    print(result[i][2])
    exit()
    image = draw_point(image, box)

    # for a, b in itertools.combinations(latent[:4], 2):
    #     relations = model.model.relation_head(torch.cat((a, b, latent[-1])))
    #     print(F.sigmoid(relations))
    #     if F.sigmoid(relations) > 0.5:
    #         p1 = model.model.box_head(a).detach().cpu().numpy()
    #         p2 = model.model.box_head(b).detach().cpu().numpy()
    #         p1[1] = 1 - p1[1]
    #         p2[1] = 1 - p2[1]
    #         p1 = p1 * 200
    #         p2 = p2 * 200
    #         p1 = p1.astype(int)
    #         p2 = p2.astype(int)
    #         print(p1, p2)
    #         cv.line(image, p1, p2, color=(255, 0, 0), thickness=2)
    attention_map = get_local.cache["Attention.forward"][0][i][0][0][:25].reshape(5, 5)
    canvas.append([image, attention_map])
visualize_attentions(canvas)
# pprint(result)
# %%
