
import import_ipynb
from vit import ViT
from sklearn.metrics import accuracy_score
from Model import *
from utility import exit

# Load MNIST dataset
transform = transforms.Compose(
    [transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)])
train_dataset = torchvision.datasets.MNIST(
    root="./data", train=True, download=True, transform=transform)
print(train_dataset[0][0].shape)
ipyplot.plot_images([train_dataset[0][0].cpu().reshape((28, 28))])
# Initialize the model, loss, and optimizer
m = ViT(
    image_size=28,
    patch_size=7,
    num_classes=10,
    dim=64,
    depth=3,
    heads=6,
    mlp_dim=16,
    channels=1,
    # pool='cls',
    # dropout=0,
    # emb_dropout=0
)
matrics = Accuracy(task="multiclass", num_classes=10)
model = Model("mnist", m, nn.CrossEntropyLoss(),
              transform=transform, eval_metrics=matrics)
# Training loop
model.fit(train_dataset, optim.Adam(m.parameters(), lr=1e-3),
          epochs=5, batch_size=64, shuffle=False, transformed=True)
