import sys

sys.path.append(".." if ("ipykernel" in sys.modules) else ".")

# rnn = CfC(20, 50)  # (input, hidden units)
# x = torch.randn(2, 3, 20)  # (batch, time, features)
# h0 = torch.zeros(2, 50)  # (batch, units)
# output, hn = rnn(x, h0)
# print(output.shape, hn.shape)
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from ncps.torch import CfC
from torch.utils.data import DataLoader

from Model import *

# Define the transformations
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# Download and load the training dataset
train_dataset = torchvision.datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Download and load the test dataset
test_dataset = torchvision.datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda()
        out, _ = self.rnn(x, h0)
        out = out[:, -1, :]
        out = self.fc(out)
        return out


class RNN_cfc(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = CfC(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda()
        # x = torch.randn(2, 3, 20)  # (batch, time, features)
        # h0 = torch.zeros(2, 50)  # (batch, units)
        hn = torch.zeros(x.size(0), self.hidden_size).cuda()
        for _ in range(self.num_layers):
            out, hn = self.rnn(x, hn)
        out = out[:, -1, :]
        out = self.fc(out)
        return out


input_size = 28  # Each row of the image is treated as a sequence of 28 time steps
hidden_size = 10  # Number of features in the hidden state
num_layers = 2  # Number of recurrent layers
num_classes = 10  # Number of output classes

model = RNN_cfc(input_size, hidden_size, num_layers, num_classes).cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 10

for epoch in range(num_epochs):
    loss_record = []
    for i, (images, labels) in tqdm(enumerate(train_loader)):
        images.cuda()
        labels.cuda()
        # Reshape the images to (batch_size, sequence_length, input_size)
        images = images.squeeze(1)
        # Move tensors to the configured device
        images = images.to("cuda")
        labels = labels.to("cuda")

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_record.append(loss.item())
    print(
        f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {np.mean(loss_record):.4f}"
    )
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.squeeze(1)
            images = images.cuda()
            labels = labels.cuda()
            # images = images.to(model.fc.weight.device)
            # labels = labels.to(model.fc.weight.device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f"Accuracy of the model on the test images: {100 * correct / total} %")
