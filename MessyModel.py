# %%
from Model import *
from createdata import MessyDataset

width = 200
d = MessyDataset(20000)
# %%
plot_images([d[i] for i in range(5)] , flatten=True, img_width=150)
# %%
transform = transforms.Compose([
    transforms.ToImage(),
    transforms.Resize((width, width)),
    transforms.ToDtype(torch.float32, scale=True),
])
model = Model("autoencoder",
              data=d, transform=transform, ytransform=transform, batch_size=256)
# %%
transforms.ToPILImage()(model.first())
# %%


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            # 200x200x1 -> 100x100x16
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            # 100x100x16 -> 50x50x32
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 50x50x32 -> 25x25x64
            nn.ReLU(),
            # 25x25x64 -> 13x13x128
            nn.Conv2d(64, 128, 3, stride=2, padding=2),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            # 13x13x128 -> 25x25x64
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1,
                               output_padding=1),  # 25x25x64 -> 50x50x32
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1,
                               output_padding=1),  # 50x50x32 -> 100x100x16
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1,
                               output_padding=1),  # 100x100x16 -> 200x200x1
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


m = Autoencoder()
model.gc()
model.fit(m, nn.MSELoss(), optim.Adam(
    m.parameters(), lr=0.001), epochs=500)
# %%
d2 = MessyDataset(10)
result = model.inference(d2)

imgs = []
for i in range(len(result)):
    imgs.append([np.array(transforms.ToPILImage()(result[0][i])), np.array(
        transforms.ToPILImage()(result[1][i])), np.array(transforms.ToPILImage()(result[2][i]))])
imgs = list(zip(*imgs))
ipyplot.plot_images(imgs[0])
ipyplot.plot_images(imgs[1])
# %%


class TestDataset(Datasetbehaviour):
    def __init__(self, size):
        self.i = 0
        self.library = ["data_cleaning_example/dac082s085-page29_SOIC_Section_0.png",
                        "data_cleaning_example/dac082s085-page29_SOIC_Short_0.png", "data_cleaning_example/dac082s085-page29_SOIC_Top_0.png"]
        super().__init__(size, self.__create)

    def __create(self):
        res = cv.imread(self.library[self.i]), cv.imread(self.library[self.i])
        self.i += 1
        return res


d3 = TestDataset(3)
result = model.inference(d3)
imgs = []
for i in range(len(result)):
    imgs.append([np.array(transforms.ToPILImage()(result[0][i])), np.array(
        transforms.ToPILImage()(result[1][i])), np.array(transforms.ToPILImage()(result[2][i]))])
imgs = list(zip(*imgs))
ipyplot.plot_images(imgs[0], img_width=200)
ipyplot.plot_images(imgs[1], img_width=200)
