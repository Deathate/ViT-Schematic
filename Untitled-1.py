# %%

from createdata import DoubleLineFormalDataset
from Model import *

d = DoubleLineFormalDataset(20)
ipyplot.plot_images([x[0][0] for x in d[:]], img_width=100, max_images=10)

# %%
import vit


class ViT_ex(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, dropout=0, channels=3, dim_head=64):
        super().__init__()
        image_height, image_width = image_size, image_size
        patch_height, patch_width = patch_size, patch_size

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        self.num_classes = num_classes
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

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

        self.transformer = vit.Transformer(
            dim, depth, heads, dim_head, mlp_dim)
        self.decoder = nn.TransformerDecoderLayer(
            d_model=dim, nhead=1, dim_feedforward=16, batch_first=True, dropout=dropout)
        self.tgt_embedding = nn.Linear(1, dim)
        self.output_net = nn.Sequential(
            # nn.Linear(64, dim),
            # nn.ReLU(),
            # nn.Linear(dim, num_classes)
            nn.Linear(128, num_classes)
        )

    def forward(self, x, y):
        img, pos, tgt = x
        img = self.to_patch_embedding(img)
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=img.shape[0])
        # pos = self.label_net(pos)
        # pos = einops.rearrange(pos, 'a (b c) -> a b c', b=1)
        img = torch.cat((cls_tokens, img), dim=1)
        img += self.pos_embedding(img)
        img = self.transformer(img)
        tgt = rearrange(tgt, 'a b c -> a (b c) 1')
        tgt = self.tgt_embedding(tgt)
        output = self.decoder(tgt, img)
        output = rearrange(output, 'a b c -> a (b c)')
        output = self.output_net(output)
        output = rearrange(output, "a (b c) -> a b c", c=2)
        # print(output.shape)
        # exit()
        return output


def transform(x): return transforms.Compose([
    transforms.ToImage(),
    transforms.Grayscale(),
    transforms.ToDtype(torch.float32, scale=True),
])(x[0]).cuda(), torch.tensor(x[1]).float().cuda(), torch.tensor(x[2]).float().cuda()


image_width = 200
patch_width = 10
m = ViT_ex(image_width, patch_width, num_classes=4, dim=32, depth=1,
           heads=1, mlp_dim=32, channels=1, dim_head=32, dropout=0.1)
model = Model("SingleLine", d, transform, cudnn=True, amp=False, batch_size=64)
get_local.deactivate()
model.fit(m, nn.MSELoss(), optim.Adam(m.parameters(), lr=0.001),
          500, target_transform=Hungarian_Order)

# %%
get_local.activate()
size = 10
testset = DoubleLineFormalDataset(size)
inference = model.inference(testset)
images = [x[0][0] for x in testset]
result = list(zip(images, *inference))
for image, inference, truth in result:
    for tr in truth:
        cv.circle(image, [(tr * 2).int().numpy()[0],
                          200 - (tr * 2).int().numpy()[1]], 8, (250, 200, 0), -1)
    for inf in inference:
        cv.circle(image, ((inf * 2).int().numpy()[
            0], 200 - (inf * 2).int().numpy()[1]), 5, (0, 255, 0), -1)
attention_maps = get_local.cache["Attention.forward"]
side_len = image_width // patch_width
maps = []
for i in tqdm.tqdm(range(size)):
    map1 = attention_maps[0][i][0][0][:side_len**2].reshape(side_len, side_len)
    map2 = attention_maps[1][i][0][0][:side_len**2].reshape(side_len, side_len)
    maps.append((images[i], map1, map2))
ipyplot.plot_images([x[0] for x in maps], img_width=120)
ipyplot.plot_images([x[1] for x in maps], img_width=120)
ipyplot.plot_images([x[2] for x in maps], img_width=120)
visualize_attentions(maps)
