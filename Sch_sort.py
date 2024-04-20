# %%
from Model import *


class SortDataset(Datasetbehaviour):
    def __init__(self, size, num_categories, n):
        super().__init__(size, self.__create, num_categories, n)

    def __create(self, num_categories, n):
        data = np.random.randint(num_categories, size=n)
        label = np.argsort(data)
        # label = np.flip(data)

        return data, label


num_categories = 20
n = 10
dataset = SortDataset(10000, num_categories, n)
dataset.view()
# %%

latent = 10


class SortModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_net = nn.Embedding(100, latent)
        self.position = PositionalEncoding(latent)
        self.network = nn.Sequential(
            nn.TransformerEncoderLayer(
                latent, 1, latent, batch_first=True), nn.Linear(latent, n),
        )

    def forward(self, x, y):
        # x = x.reshape(-1, 10, 1)
        x = self.input_net(x)
        x = x + self.position(x)
        x = self.network(x)
        return x


def xtransform(x):
    return torch.tensor(x, dtype=torch.long).cuda()


def ytransform(x):
    return torch.tensor(x, dtype=torch.long).cuda()


model = Model("", dataset, 256, xtransform=xtransform, ytransform=ytransform)
m = SortModel()


def criterion(y_hat, y):
    y_hat = einops.rearrange(
        y_hat, "a b c -> (a b) c")
    y = einops.rearrange(y, "a b -> (a b)")
    return F.cross_entropy(y_hat, y)


model.fit(m, criterion, optim.Adam(m.parameters(), 1e-3), 2000)
# %%
test = SortDataset(1, num_categories, n)
result = model.inference(test)
# print(result[0][1].argmax(1))
