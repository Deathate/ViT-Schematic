# %%
from Model import *


class ReverseDataset(Datasetbehaviour):
    def __init__(self, size, num_categories, seq_len):
        super().__init__(size, self.__create, num_categories, seq_len)

    def __create(self, num_categories, seq_len):
        data = rng.integers(
            num_categories, size=(seq_len))
        labels = np.flip(data)

        return [data.tolist(), labels]


d = ReverseDataset(50000, 10, 16)
d.view()

# %%

transform = transforms.Compose([
    lambda x: F.one_hot(x, 10).float(),
])
from vit import Transformer


class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.position = PositionalEncoding(32)
        self.input_net = nn.Sequential(
            nn.Linear(10, 32),
        )
        self.mlp = nn.Sequential(
            Transformer(32, 1, 1, dim_head=64, mlp_dim=32),
            nn.Linear(32, 10),
        )

    def forward(self, x):
        x = self.input_net(x)
        x = x + self.position(x)
        x = self.mlp(x)
        return x


m = M()


def criterion(y_hat, y): return F.cross_entropy(einops.rearrange(
    y_hat, "a b c -> (a b) c"), einops.rearrange(y, "a b -> (a b)"))


model = Model("ReverseDataset", m, criterion, transform=transform)
model.fit(d, optimizer=optim.Adam(model.parameters(), lr=5e-4),
          epochs=10, batch_size=128, validation_split=0.1)
# torchmetrics.functional.accuracy(model(d[0]), d[1])
# MulticlassAccuracy()

# %%
get_local.clear()

# %%
print(len(get_local.cache["Attention.forward"]))

# %%
print(model.predict(d[:10][0]).shape)
print(d[:10][1].shape)
y_hat = model.predict(d[:10][0]).cpu()
y = d[:10][1]
MulticlassAccuracy(num_classes=10)(einops.rearrange(
    y_hat, "a b c -> (a b) c"), einops.rearrange(y, "a b -> (a b)")).item()

# %%
attention_maps = get_local.cache['Attention.forward']
# ipyplot.plot_images(attention_maps[0][0])
# display(Image.fromarray(attention_maps[0][0].view(16, 16)))
ipyplot.plot_images([attention_maps[i][0].reshape(16, 16) for i in range(5)])
