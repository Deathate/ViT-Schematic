# %%
from scipy.spatial.distance import cdist

import config
from Model import *
from vit import Transformer
# set_seed(0)

class FormalDatasetWindowed(Datasetbehaviour):
    def __init__(self, size=None):
        self.dataset_folder = Path("dataset_windowed/pkl")
        self.dataset_list = list(self.dataset_folder.iterdir())
        if size is None:
            size = len(self.dataset_list)
        self.dataset_list = self.dataset_list[:size]
        self.i = 0
        self.num_empty = 0
        super().__init__(size, self.__create)

    def __create(self):
        path = self.dataset_list[self.i]
        data = pickle.load(open(path, "rb"))
        img = cv2.imread("dataset_windowed/images/" + Path(path).stem + ".png", cv2.IMREAD_UNCHANGED)
        img_height, img_width = img.shape[:2]
        for net, prop in data.items():
            prop = [x for x in prop if len(x) == 2 or (len(prop)==1 and len(x)==1)]
            try:
                data[net] = np.array(prop).reshape(-1, 4).astype(np.float32).round(4)
                data[net][:, [0, 2]] = data[net][:, [0, 2]] / img_width
                data[net][:, [1, 3]] = data[net][:, [1, 3]] / img_height
            except:
                data[net] = np.array(prop).reshape(-1, 2).astype(np.float32).round(4)
                data[net][:, 0] = data[net][:, 0] / img_width
                data[net][:, 1] = data[net][:, 1] / img_height
        lines = []
        for net, prop in data.items():
            try:
                line = np.array(prop).reshape(-1, 2, 2)
                lines.append(line)
            except:
                pass
        for net, prop in data.items():
            data[net] = np.array(data[net]).reshape(-1, 2)
        points = np.vstack(list(data.values())).reshape(-1, 2)
        points = np.unique(points, axis=0)

        # if self.i == 3:
        #     plot_images(img, img_width=img_width)
        #     plot_images(draw_point(img, points), img_width=img_width)
        #     plot_images(draw_lines(img, lines), img_width=img_width)
        #     print(np.array(sorted(lines[0], key=lambda x: (x[0][0], x[1][0]))))
        #     print(path)
        #     print(len(points))
        #     exit()
        self.i += 1
        if len(points) == 0:
            self.num_empty += 1
        return img, points

Datasetbehaviour.MP = False
dataset_guise = FormalDatasetWindowed(config.DATASET_SIZE)
dataset_guise.view()
# print(dataset_guise.num_empty)
# print(len(dataset_guise))
# for i in range(10):
#     plot_images(
#         draw_point(dataset_guise[i][0], dataset_guise[i][1]), img_width=dataset_guise[i][0].shape[0]
#     )
print(dataset_guise.num_empty)
# in 104755 images, 28353 of images is almost empty
result_num = 0
for x in dataset_guise:
    result_num = max(x[1].shape[0], result_num)
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
if config.EVAL:
    for i in range(3):
        plot_images(draw_point(dataset_guise[i][0], dataset_guise[i][1]), img_width=100)
        plot_images(dataset_guise[i][0], img_width=100)
    dataset_guise.view()


# %%
def xtransform(x):
    gray_channel=transforms.Compose(
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
    eval = config.EVAL,
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
        # self.label_query = nn.Sequential(
        #     nn.Linear(1, dim),
        #     nn.LayerNorm(dim),
        # )

    def forward(self, x, y):
        x = x.to(torch.float32)
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
    Hungarian_Order(predicted_box, y)
    gtruth_label = y[:, :, 0] >= 0
    valid_label = gtruth_label.any(dim=1)
    gtruth_label = gtruth_label.to(torch.float32)
    predicted_box[predicted_label] = y[predicted_label].to(predicted_box.dtype)

    empty_weight = 1
    loss_box = 0
    for i in range(len(y)):
        value = F.smooth_l1_loss(predicted_box[i], y[i])
        if valid_label[i]:
            loss_box += value
        else:
            loss_box += empty_weight * value
    loss_box = loss_box / len(y)
    # loss_box = F.smooth_l1_loss(predicted_box, y)
    loss_class = F.binary_cross_entropy_with_logits(predicted_label_logit, gtruth_label)
    if stage == 0:
        return 3 * loss_box + loss_class
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
                if meta.data[i].meta.has_edge(c1n, c2n):
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
                    # joint_token = torch.cat((obj_token_1, obj_token_2))
                    # joint_token_inv = torch.cat((obj_token_2, obj_token_1))
                    joint_token_list.append(joint_token)
                    joint_token_list.append(joint_token_inv)
                    label_list.append(label)
                    label_list.append(label)
        loss_relation = F.binary_cross_entropy_with_logits(
            meta.model.relation_heads(torch.stack(joint_token_list)),
            torch.stack(label_list).cuda(),
        )
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


style = "decoder"
# m = ViT_ex(image_size=200, patch_size=patch_size, dim=dim, depth=1,
#            heads=8, mlp_dim=32, channels=1, dim_head=32, dropout=0.1, result_num=15, style=style)
head_num = 8
dim_head = 50
m = ViT_ex(
    image_size=100,
    patch_size=10,
    dim=head_num * dim_head,
    depth=6,
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
    optim.AdamW(m.parameters(), lr=config.LEARNING_RATE),
    config.EPOCHS,
    max_epochs=config.MAX_EPOCHS if hasattr(config, "MAX_EPOCHS") else float("inf"),
    pretrained_path=config.PRETRAINED_PATH,
    keep=not config.EVAL,
    backprop_freq=config.BATCH_STEP,
    device_ids=config.DEVICE_IDS,
    keep_epoch=config.KEEP_EPOCH,
)
# %%
# Datasetbehaviour.RESET = True
if config.EVAL:
    dataset_test = FormalDatasetWindowed(10)
    result = model.inference(dataset_test)

    # print(model.model.box_head(result[0][1][:4]))
    # print(model.model.relation_head(torch.cat((result[0][1][0], result[0][1][1], result[0][1][-1]))))
    canvas = []
    for i, d in enumerate(dataset_test):
        image = dataset_test[i][0]
        image_bk = image.copy()
        image_bk = draw_point(image_bk, dataset_test[i][1])
        image_line = image.copy()
        latent = result[i][1]
        box_latent = model.model.box_head(latent[:result_num])
        box = box_latent[:, :2]
        classes = F.sigmoid(box_latent[:, 2]) > 0.5
        box[~classes] = 0
        image = draw_point(image, box)
        # for a, b in itertools.combinations(latent[:result_num], 2):
        #     relations = model.model.relation_heads(torch.cat((a, b, latent[-1])))
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
        #             color = random.choice([(255, 0, 0), (0, 255, 0), (0, 0, 255)])
        #             cv2.line(image_line, p1_pos, p2_pos, color=(0, 0, 255), thickness=2)
        # attention_map = get_local.cache["Attention.forward"][0][i][0][0][:25].reshape(5, 5)
        canvas.append([image_bk, image, image_line])
    visualize_attentions(canvas)
    # pprint(result)
