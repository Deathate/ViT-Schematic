import itertools

from Model import *


class WindowedRelation(Datasetbehaviour):
    def __init__(self, size_one):
        self.i = -1
        data = {"0": [], "1": []}
        for i in itertools.count():
            d = pickle.load(open(f"dataset_relation_latent/{i+1}.pkl", "rb"))
            data["0"].extend(d["0"])
            data["1"].extend(d["1"])
            if len(data["1"]) > size_one:
                break
        self.data = data["0"][: size_one * 2] + data["1"][:size_one]
        random.shuffle(self.data)
        size = len(self.data)
        # label 0: 4221286 label 1: 962588
        super().__init__(size, self.__create)

    def __create(self):
        self.i += 1
        return self.data[self.i][0], self.data[self.i][1]


dataset = WindowedRelation(100000)
# dataset.view()


def xtransform(x):
    return torch.tensor(x).float()


def ytransform(x):
    return torch.tensor(x).float()


model = Model(
    "WindowedRelation",
    dataset,
    xtransform=xtransform,
    ytransform=ytransform,
    amp=True,
    batch_size=256,
)
model.view()


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, num_classes, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.to_logits = nn.Sequential(nn.Linear(hidden_dim, num_classes))

    def forward(self, x):
        return self.to_logits(self.net(x))


m = MLP(dim=1200, hidden_dim=400, num_classes=1, dropout=0)


def criterion(y_hat, y):
    # print(y_hat.shape)
    # print(y.shape)
    # exit()
    loss = F.binary_cross_entropy_with_logits(y_hat.flatten(), y)
    return loss


def eval_metrics(criterion, y_hat, y):
    loss = criterion(y_hat, y)
    length = y_hat.shape[0]
    accs = 0
    for i in range(length):
        label = 1 if y_hat[i].sigmoid() >= 0.5 else 0
        if label == y[i]:
            accs += 1
    accs = accs / length
    accs = torch.tensor(accs)
    return loss, accs


model.fit(
    m,
    criterion,
    optim.AdamW(m.parameters(), lr=0.001),
    500,
    max_epochs=500,
    eval_metrics=eval_metrics,
)
