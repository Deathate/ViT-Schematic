# %%
from scipy.spatial.distance import cdist

import config
from Model import *
from vit import Transformer

if config.DETERMINISTIC:
    set_seed(0, True)
else:
    set_seed(0, False)


class FormalDatasetWindowed(Datasetbehaviour):
    def __init__(self, size=None):
        self.dataset_source = config.DATASET_PATH
        self.dataset_folder = Path(self.dataset_source) / Path("pkl")
        self.dataset_list = list(self.dataset_folder.iterdir())
        random.shuffle(self.dataset_list)
        if size is None:
            size = len(self.dataset_list)
        self.dataset_list = self.dataset_list[:size]
        self.i = 0
        super().__init__(size, self.__create)

    def __create(self):
        path = self.dataset_list[self.i]
        self.i += 1
        data = pickle.load(open(path, "rb"))
        img = cv2.imread(
            self.dataset_source + "/images/" + Path(path).stem + ".png", cv2.IMREAD_UNCHANGED
        )
        # if img.sum() / 255 / 10000 / 4 > 0.99:
        #     return None
        img_height, img_width = img.shape[:2]
        for net, prop in data.items():
            prop = [x for x in prop if len(x) == 2 or (len(prop) == 1 and len(x) == 1)]
            try:
                data[net] = np.array(prop).reshape(-1, 4).astype(np.float32).round(4)
                data[net][:, [0, 2]] = data[net][:, [0, 2]] / img_width
                data[net][:, [1, 3]] = data[net][:, [1, 3]] / img_height
            except:
                data[net] = np.array(prop).reshape(-1, 2).astype(np.float32).round(4)
                data[net][:, 0] = data[net][:, 0] / img_width
                data[net][:, 1] = data[net][:, 1] / img_height
        lines = []
        G = nx.Graph()
        for net, prop in data.items():
            try:
                line = np.array(prop).reshape(-1, 2, 2)
                if line.size > 0:
                    lines.append(line)
            except:
                pass
        if len(lines) > 0:
            lines = np.vstack(lines)
            for line in lines:
                G.add_edge(tuple(line[0]), tuple(line[1]))
        # print(lines)
        # print(G.has_edge(np.array(lines[0][1]).tobytes(), np.array(lines[0][0]).tobytes()))
        # exit()
        for net, prop in data.items():
            data[net] = np.array(data[net]).reshape(-1, 2)
        points = np.vstack(list(data.values())).reshape(-1, 2)
        points = np.unique(points, axis=0)

        # if self.i < 5:
        #     plot_images(img, img_width=img_width)
        #     plot_images(draw_point(img, points), img_width=img_width)
        #     plot_images(draw_lines(img, lines), img_width=img_width)
        #     print(points)
        #     print(path)
        #     print(len(points))
        # else:
        #     exit()

        return img, points, G


Datasetbehaviour.MP = False
if config.EVAL:
    Datasetbehaviour.RESET = True
dataset_guise = FormalDatasetWindowed(config.DATASET_SIZE)
if config.EVAL:
    dataset_guise.view()
# print(dataset_guise.num_empty)
# print(len(dataset_guise))
# for i in range(10):
#     plot_images(
#         draw_point(dataset_guise[i][0], dataset_guise[i][1]), img_width=dataset_guise[i][0].shape[0]
#     )

# in 104755 images, 28353 of images is almost empty
# result_num = 0
# for x in dataset_guise:
#     result_num = max(x[1].shape[0], result_num)
result_num = 36
# %%
# for max_ele in random.choices(L, k=10):
#     path =max_ele[1]
#     points = max_ele[2]
#     img  = cv.imread("dataset/images/"+path.stem + ".jpg")
#     img = draw_point(img, points,width=2)
#     plot_images([img], img_width=600)
# print(points)
# print(path)
# print(len(points))

# %%
# if config.EVAL:
#     for i in range(3):
#         plot_images(draw_point(dataset_guise[i][0], dataset_guise[i][1]), img_width=100)
#         plot_images(dataset_guise[i][0], img_width=100)
#     dataset_guise.view()


# %%
def xtransform(x):
    gray_channel = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.Grayscale(),
            transforms.ToDtype(torch.float32, scale=True),
        ]
    )(x[:, :, :3])
    alpha_channel = x[:, :, -1] / 255
    alpha_channel = alpha_channel[np.newaxis, :]
    alpha_channel = torch.tensor(alpha_channel)
    joint = torch.cat((gray_channel, alpha_channel), dim=0)
    return joint


def ytransform(x):
    s = np.full((result_num, 2), -1, dtype=np.float32)
    s[: len(x)] = x
    return torch.tensor(s).float()


model = Model(
    "FormalDatasetWindowed",
    dataset_guise,
    xtransform=xtransform,
    ytransform=ytransform,
    amp=True,
    batch_size=config.BATCH_SIZE,
    eval=config.EVAL,
    config={
        x: getattr(config, x)
        for x in dir(config)
        if not (x.startswith("_") or inspect.ismodule(getattr(config, x)))
    },
    # memory_fraction=0.5,
)
# model.view()
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
        self.relation_heads = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 1),
        )
        # self.cnn = nn.Conv2d(2, 1, 3, padding=1)
        # self.label_query = nn.Sequential(
        #     nn.Linear(1, dim),
        #     nn.LayerNorm(dim),
        # )

    def forward(self, x, y):
        x = x.to(torch.float32)
        # x = self.cnn(x)
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
        elif self.style == "decoder_cls":
            cls_tokens = repeat(self.cls_token, "1 1 d -> b 1 d", b=x.shape[0])
            x = torch.cat((cls_tokens, x), dim=1)
            x += self.pos_embedding(x)
            x = self.transformer(x)
            cls_latent = x[:, :1]
            tgt = repeat(self.decoder_query.weight, "d e -> n d e", n=x.shape[0])
            x = self.decoder(tgt, cls_latent)
            # x = self.mlp(x)
            # x = self.box_head(x)
            return x, cls_latent

        elif self.style == "decoder_only":
            # xpos = y[:, :, :1]
            # xpos = self.label_query(xpos)
            x += self.pos_embedding(x)
            # x = torch.cat((x, xpos), dim=1)

            tgt = repeat(self.decoder_query.weight, "d e -> n d e", n=x.shape[0])
            x = self.decoder(tgt, x)
        else:
            raise ValueError("style not found")
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


stage = config.STAGE
ts = 1


def criterion(y_hat, y):
    if isinstance(y_hat, tuple):
        y_hat, cls_latnt = y_hat
    box_head_result = model.meta.model.box_head(y_hat[:, :result_num])
    predicted_box, predicted_label_logit = box_head_result[:, :, :2], box_head_result[:, :, 2]
    # predicted_label = torch.sigmoid(predicted_label_logit) < 0.5
    Hungarian_Order(predicted_box, y)
    # gtruth_label = y[:, :, 0] > -0.01
    # gtruth_label = gtruth_label.to(torch.float32)
    # predicted_box[predicted_label] = y[predicted_label].to(predicted_box.dtype)
    # empty_weight = 0.1
    # loss_box = []
    # for i in range(len(y)):
    #     value = F.smooth_l1_loss(predicted_box[i], y[i])
    #     if y[i].any():
    #         loss_box.append(value)
    #     else:
    #         loss_box.append(value * empty_weight)
    # loss_box = torch.stack(loss_box).mean()
    loss_box = F.smooth_l1_loss(predicted_box, y)
    if stage == 0:
        return loss_box, predicted_box
    # loss_class = F.binary_cross_entropy_with_logits(predicted_label_logit, gtruth_label)
    # if stage == 0:
    #     return 3 * loss_box + 0 * loss_class
    elif stage == 1:
        valid_relation_num = 1
        non_valid_relation_num = 2
        joint_token_list = []
        label_list = []
        for i, element in enumerate(y.cpu().detach().numpy()):
            valid_relation = []
            non_valid_relation = []
            element_with_order = list(enumerate(element))
            np.random.shuffle(element_with_order)
            for c1, c2 in itertools.combinations(element_with_order, 2):
                c1n = c1[1].tobytes()
                c2n = c2[1].tobytes()
                box_pair = [c1[0], c2[0]]
                if c1[1][0] >= 0 and c2[1][0] >= 0:
                    if model.meta.data[i].meta.has_edge(c1n, c2n):
                        valid_relation.append(box_pair)
                    else:
                        non_valid_relation.append(box_pair)
            for times, relation, label in [
                (min(valid_relation_num, len(valid_relation)), valid_relation, 1),
                (min(non_valid_relation_num, len(non_valid_relation)), non_valid_relation, 0),
            ]:
                label = torch.tensor([label], dtype=torch.float32)
                for time in range(times):
                    relation_from = relation[time][0]
                    relation_to = relation[time][1]
                    obj_token_1 = y_hat[i][relation_from].detach()
                    obj_token_2 = y_hat[i][relation_to].detach()
                    joint_token = torch.cat((obj_token_1, obj_token_2))
                    joint_token_inv = torch.cat((obj_token_2, obj_token_1))
                    joint_token_list.append(joint_token)
                    joint_token_list.append(joint_token_inv)
                    label_list.append(label)
                    label_list.append(label)
        loss_relation = F.binary_cross_entropy_with_logits(
            model.meta.model.relation_heads_simple(torch.stack(joint_token_list)),
            torch.stack(label_list).cuda(),
        )
        return loss_box + loss_relation, predicted_box
    elif stage == 2:
        valid_relation_num = 3
        non_valid_relation_num = 3
        joint_token_list = []
        label_list = []
        for i, element in enumerate(y.cpu().detach().numpy()):
            valid_relation = []
            non_valid_relation = []
            element_with_order = list(enumerate(element))
            np.random.shuffle(element_with_order)
            for (c1idx, c1), (c2idx, c2) in itertools.combinations(element_with_order, 2):
                if c1[0] < 0 or c1[1] < 0 or c2[0] < 0 or c2[1] < 0:
                    continue
                box_pair = [c1idx, c2idx]
                c1n = tuple(c1)
                c2n = tuple(c2)
                if model.meta.data[i].meta.has_edge(c1n, c2n):
                    valid_relation.append(box_pair)
                else:
                    non_valid_relation.append(box_pair)
                if (
                    len(valid_relation) >= valid_relation_num
                    and len(non_valid_relation) >= non_valid_relation_num
                ):
                    break
            for times, relation, label in (
                (min(valid_relation_num, len(valid_relation)), valid_relation, 1),
                (min(non_valid_relation_num, len(non_valid_relation)), non_valid_relation, 0),
            ):
                label = torch.tensor([label], dtype=torch.float32)
                for time in range(times):
                    relation_from = relation[time][0]
                    relation_to = relation[time][1]
                    obj_token_1 = y_hat[i][relation_from]
                    obj_token_2 = y_hat[i][relation_to]
                    rln_token = y_hat[i][-1]
                    joint_token = torch.cat((obj_token_1, obj_token_2, rln_token))
                    joint_token_list.append(joint_token)
                    label_list.append(label)
                    joint_token_inv = torch.cat((obj_token_2, obj_token_1, rln_token))
                    joint_token_list.append(joint_token_inv)
                    label_list.append(label)
        loss_relation = F.binary_cross_entropy_with_logits(
            model.meta.model.relation_heads(torch.stack(joint_token_list)),
            torch.stack(label_list).cuda(),
        )
        return 3 * loss_box + loss_relation, predicted_box
    elif stage == -1:
        temp = {"0": [], "1": []}
        joint_token_list = []
        label_list = []
        for i, element in enumerate(y.cpu().detach().numpy()):
            valid_relation = []
            non_valid_relation = []
            element_with_order = list(enumerate(element))
            np.random.shuffle(element_with_order)
            for c1, c2 in itertools.combinations(element_with_order, 2):
                c1n = c1[1].tobytes()
                c2n = c2[1].tobytes()
                box_pair = [c1[0], c2[0]]
                if c1[1][0] >= 0 and c2[1][0] >= 0:
                    if model.meta.data[i].meta.has_edge(c1n, c2n):
                        valid_relation.append(box_pair)
                    else:
                        non_valid_relation.append(box_pair)
            for times, relation, label in [
                (len(valid_relation), valid_relation, 1),
                (len(non_valid_relation), non_valid_relation, 0),
            ]:
                label_t = torch.tensor([label], dtype=torch.float32)
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
                    label_list.append(label_t)
                    label_list.append(label_t)
                    temp["1" if label == 1 else "0"].append(
                        (
                            torch.cat((cls_latent[i][0], obj_token_1, obj_token_2))
                            .detach()
                            .cpu()
                            .numpy(),
                            label,
                        )
                    )
                    temp["1" if label == 1 else "0"].append(
                        (
                            torch.cat((cls_latent[i][0], obj_token_2, obj_token_1))
                            .detach()
                            .cpu()
                            .numpy(),
                            label,
                        )
                    )
        if model.meta.mode == "train":
            global ts
            with open(f"dataset_relation_latent/{ts}.pkl", "wb") as f:
                pickle.dump(temp, f)
            ts += 1
        if model.meta.epoch == 1:
            exit()
        loss_relation = F.binary_cross_entropy_with_logits(
            model.meta.model.relation_heads(torch.stack(joint_token_list)),
            torch.stack(label_list).cuda(),
        )
    return loss_relation, predicted_box


def eval_metrics(criterion, y_hat, y):
    loss, pbox = criterion(y_hat, y)
    with torch.no_grad():
        length = pbox.shape[0]
        accs = 0
        for i in range(length):
            p1s = pbox[i]
            p2s = y[i]
            dm = torch.cdist(p1s, p2s, p=1)
            dm = torch.diag(dm)
            acc = (dm < 0.05).float().mean()
            accs += acc
        accs = accs / length

        # joint_token_list = []
        # label_list = []
        # for i, element in enumerate(y.cpu().numpy()):
        #     valid_relation = []
        #     non_valid_relation = []
        #     element_with_order = list(enumerate(element))
        #     for (c1idx, c1), (c2idx, c2) in itertools.combinations(element_with_order, 2):
        #         if c1[0] < 0 or c1[1] < 0 or c2[0] < 0 or c2[1] < 0:
        #             continue
        #         box_pair = [c1idx, c2idx]
        #         c1n = tuple(c1)
        #         c2n = tuple(c2)
        #         if model.meta.data[i].meta.has_edge(c1n, c2n):
        #             valid_relation.append(box_pair)
        #         else:
        #             non_valid_relation.append(box_pair)
        #     for times, relation, label in (
        #         (len(valid_relation), valid_relation, True),
        #         (len(non_valid_relation), non_valid_relation, False),
        #     ):
        #         label = torch.tensor([label], dtype=torch.float32)
        #         for time in range(times):
        #             relation_from = relation[time][0]
        #             relation_to = relation[time][1]
        #             obj_token_1 = y_hat[i][relation_from]
        #             obj_token_2 = y_hat[i][relation_to]
        #             rln_token = y_hat[i][-1]
        #             joint_token = torch.cat((obj_token_1, obj_token_2, rln_token))
        #             joint_token_list.append(joint_token)
        #             label_list.append(label)
        #             joint_token_inv = torch.cat((obj_token_2, obj_token_1, rln_token))
        #             joint_token_list.append(joint_token_inv)
        #             label_list.append(label)
        # joint_token_list = torch.stack(joint_token_list)
        # label_hat = (model.model.relation_heads(joint_token_list).sigmoid() > 0.5).flatten().cpu()
        # label_gt = torch.stack(label_list).flatten().to(torch.bool)
        # label_acc = (label_hat == label_gt).float().mean()
        # wandb.log({"acc": accs, "loss": loss, "label_acc": label_acc})
    return loss, accs


style = config.MODEL_STYLE
depth = config.DEPTH
head_num = config.NUM_HEADS
dim_head = config.EMBED_DIM
m = ViT_ex(
    image_size=100,
    patch_size=config.PATCH_SIZE,
    dim=head_num * dim_head,
    depth=depth,
    heads=head_num,
    dim_ff=head_num * dim_head,
    result_num=result_num,
    channels=2,
    dropout=config.DROPOUT,
    style=style,
)
model.suffix = "_" + style
model.fit(
    m,
    criterion,
    optim.AdamW(m.parameters(), lr=config.LEARNING_RATE if stage == 0 else config.LEARNING_RATE),
    config.EPOCHS,
    max_epochs=config.MAX_EPOCHS if hasattr(config, "MAX_EPOCHS") else float("inf"),
    pretrained_path=config.PRETRAINED_PATH,
    keep=not config.EVAL,
    backprop_freq=config.BATCH_STEP,
    device_ids=config.DEVICE_IDS,
    eval_metrics=eval_metrics,
    keep_epoch=config.KEEP_EPOCH,
    keep_optimizer=config.KEEP_OPTIMIZER,
)
# %%
if config.EVAL:
    with torch.no_grad():
        # Datasetbehaviour.RESET = True
        dataset_test = FormalDatasetWindowed(100)
        result = model.inference(dataset_test)
        # print(model.model.box_head(result[0][1][:4]))
        # print(model.model.relation_head(torch.cat((result[0][1][0], result[0][1][1], result[0][1][-1]))))
        canvas = []
        k = 3
        for i in range(k, k + 10):
            image = dataset_test[i][0][:, :, :3]
            # image_bk = image.copy()
            latent = result[i][1]
            box_latent = model.model.box_head(latent[:result_num])
            box = box_latent[:, :2].cpu().numpy()
            # classes = F.sigmoid(box_latent[:, 2]) > 0.5
            # box[~classes] = 0
            image_point = draw_point(image.copy(), box, width=2, color=(255, 0, 0))
            image_point_gt = draw_point(
                image.copy(), dataset_test[i][1], width=2, color=(255, 0, 0)
            )
            # print(box[np.where(box[:, 0] > 0)])
            image_line = image.copy()
            for a, b in itertools.combinations(latent[:result_num], 2):
                relations = model.model.relation_heads(torch.cat((a, b, latent[-1])))
                if F.sigmoid(relations) > 0.5:
                    p1 = model.model.box_head(a).detach().cpu()[:2]
                    p2 = model.model.box_head(b).detach().cpu()[:2]
                    if all(p1 >= 0) and all(p2 >= 0):
                        p1_pos, p2_pos = p1.numpy(), p2.numpy()
                        p1_pos[1] = 1 - p1_pos[1]
                        p2_pos[1] = 1 - p2_pos[1]
                        p1_pos *= 100
                        p2_pos *= 100
                        p1_pos = p1_pos.astype(int)
                        p2_pos = p2_pos.astype(int)
                        color = random.choice([(255, 0, 0), (0, 255, 0), (0, 0, 255)])
                        cv2.line(image_line, p1_pos, p2_pos, color=(0, 255, 0), thickness=2)
            # print(box[(box[:, 0] > 0) & (box[:, 1] > 0)])
            # exit()
            # exit()
            # attention_map = get_local.cache["Attention.forward"][0][i][0][0][:25].reshape(5, 5)
            # canvas.append([image, image_point, image_point_gt, image_line])
            image_line = draw_point(image_line, box, width=2, color=(255, 0, 0))
            canvas.append([image, image_point_gt, image_point, image_line])
        visualize_attentions(canvas)
    # pprint(result)
