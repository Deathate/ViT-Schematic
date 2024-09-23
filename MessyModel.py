# %%
from createdata import TestDataset
from MessyDataset import MessyDataset
from Model import *

width = 200
# Datasetbehaviour.RESET = True
Datasetbehaviour.MP = True
ds = MessyDataset(10000)

plot_images(ds)

# %%
xtransform = transforms.Compose([
    transforms.ToImage(),
    transforms.Grayscale(),
    transforms.Resize((width, width)),
    transforms.ToDtype(torch.float32, scale=True),
])


ytransform = transforms.Compose(
    [
        transforms.ToImage(),
        transforms.Grayscale(),
        transforms.Resize((width, width)),
        transforms.ToDtype(torch.float32, scale=True),
    ])


model = Model("autoencoder", ds, 512, xtransform, ytransform, shuffle=False)
plot_images(model, 200, max_images=2)

# %%


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            # 200x200x1 ->
            # 100x100x16
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            # 50x50x16
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            # 25x25x32
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            # 13x13x64
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            # 25x25x32
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            # 50x50x16
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            # 100x100x16
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            # 200x200x1
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x, y):
        x = self.encoder(x)
        x = self.decoder(x)

        return x


class TransformerAutoencoder(nn.Module):
    def __init__(self, image_size, patch_size, dim, heads, mlp_dim, dropout=0, channels=3):
        super().__init__()
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
        self.to_patch_debedding = nn.Sequential(
            nn.Linear(dim, patch_dim),
            nn.Sigmoid(),
            # nn.ReLU(),
            Rearrange(
                "b (h w) (p1 p2 c) -> b c (h p1) (w p2)",
                p1=patch_height,
                p2=patch_width,
                h=image_height // patch_height,
                w=image_width // patch_width,
            ),
        )

        self.pos_embedding = PositionalEncoding(dim)

        self.encoder = nn.TransformerEncoderLayer(
            dim, heads, mlp_dim, batch_first=True, dropout=dropout
        )
        self.decoder = nn.TransformerDecoderLayer(
            dim, heads, mlp_dim, batch_first=True, dropout=dropout
        )
        self.tgt = nn.Parameter(torch.randn(1, patch_dim, dim))

    def forward(self, x, target):
        x = self.to_patch_embedding(x)
        x += self.pos_embedding(x)
        x = self.encoder(x)
        tgt = repeat(self.tgt, '1 a b -> c a b', c=x.shape[0])
        x = self.decoder(x, tgt)
        x = self.to_patch_debedding(x)
        return x


# m = Autoencoder()
m = TransformerAutoencoder(
    width, patch_size=10, dim=50, heads=1, mlp_dim=80, dropout=0, channels=1
)

model.fit(m, loss_func(nn.MSELoss()), optim.Adam(m.parameters(), lr=1e-3), epochs=1000)

# %%
# Datasetbehaviour.RESET = False
ts = MessyDataset(10)
result = model.inference(ts)
plot_images(result, img_width=200, max_images=10)
result = model.inference(TestDataset(3))
plot_images(result, img_width=300)
