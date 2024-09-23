# %%
from mambapy.mamba import Mamba, MambaConfig

from Model import *


class ReverseDataset(Datasetbehaviour):
    def __init__(self, size, num_categories, seq_len):
        super().__init__(size, self.__create, num_categories, seq_len)

    def __create(self, num_categories, seq_len):
        data = rng.integers(num_categories, size=(seq_len))
        labels = np.flip(data)

        return [data.tolist(), labels]


d = ReverseDataset(50000, 10, 16)
d.view()


# %%


from vit import Transformer


def transform(x):
    x = torch.tensor(x)
    return F.one_hot(x, 10).float()


class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.position = PositionalEncoding(32)
        self.input_net = nn.Sequential(
            nn.Linear(10, 32),
        )
        self.mlp = nn.Sequential(
            Mamba(MambaConfig(d_model=32, n_layers=5)),
            # Transformer(32, 1, 1, dim_head=64, mlp_dim=32),
            nn.Linear(32, 10),
        )

    def forward(self, x):
        # x = self.input_net(x)
        # x += self.position(x)
        x = self.mlp(x)
        return x


# test F.cross_entropy
# y_hat = torch.randn(2, 3, 5).float()
# y = torch.empty(2, 3, dtype=torch.long).random_(5)
# F.cross_entropy(einops.rearrange(y_hat, "a b c -> (a b) c"), einops.rearrange(y, "a b -> (a b)"))
# exit()
m = M()


def criterion(y_hat, y):
    return F.cross_entropy(
        einops.rearrange(y_hat, "a b c -> (a b) c").float(),
        einops.rearrange(y, "a b -> (a b)").long(),
    )


model = Model("ReverseDataset", d, xtransform=transform)
model.fit(
    m,
    criterion=criterion,
    optimizer=optim.Adam(m.parameters(), lr=5e-4),
    epochs=10,
)
# torchmetrics.functional.accuracy(model(d[0]), d[1])
# MulticlassAccuracy()

# %%
# get_local.clear()
# print(len(get_local.cache["Attention.forward"]))
# %%
d2 = ReverseDataset(10, 10, 16)
y_hat = model.inference(d2)
print(torch.argmax(column(y_hat, 1)[0], axis=1))
print(column(y_hat, 2)[0])
# print(np.argmax(column(y_hat, 1), axis=2))
# MulticlassAccuracy(num_classes=10)(
#     einops.rearrange(y_hat, "a b c -> (a b) c"), einops.rearrange(y, "a b -> (a b)")
# ).item()

# %%
attention_maps = get_local.cache["Attention.forward"]
# ipyplot.plot_images(attention_maps[0][0])
# display(Image.fromarray(attention_maps[0][0].view(16, 16)))
ipyplot.plot_images([attention_maps[i][0].reshape(16, 16) for i in range(5)])
