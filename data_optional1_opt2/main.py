import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from matplotlib import pyplot as plt
import numpy as np

import random

from dataset import get_data

X_train = get_data('dataset')
X_train = torch.from_numpy(X_train)
X_dataset = TensorDataset(X_train)
X_dataloader = DataLoader(X_dataset, batch_size=64, shuffle=True)

lcode = 64


device = "cpu"

class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.shrink = nn.Sequential(
            nn.Conv2d(3, 16, (3, 3),  padding=1),
            nn.MaxPool2d((2, 2)),
            nn.ReLU(),
            nn.Conv2d(16, 64, (3, 3), padding=1),
            nn.MaxPool2d((2, 2)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*8*8, 64*8),
            nn.ReLU()
        )
        self.mean = nn.Linear(64*8, lcode)
        self.logvar = nn.Linear(64*8, lcode)
        self.decode = nn.Sequential(
            nn.Linear(lcode, 64*8),
            nn.ReLU(),
            nn.Linear(64*8, 64*8*8),
            nn.ReLU(),
            nn.Unflatten(1, (64, 8, 8)),
            nn.ConvTranspose2d(64, 16, (3, 3), stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, (4, 4), stride=2),
            nn.Sigmoid()
        )

    def sample(self, mean, logvar):
        return mean + torch.exp(logvar / 2) * torch.randn_like(mean)

    def encode(self, x):
        y = self.shrink(x)
        return self.sample(self.mean(y), self.logvar(y))
    
    def forward(self, x):
        y = self.shrink(x)
        m = self.mean(y)
        v = self.logvar(y)
        return (m, v, self.decode(self.sample(m, v)))

model = VAE().to(device)
print(model)

def kl_std(mean, logvar):
    return torch.sum(torch.exp(logvar) + mean * mean - 1 - logvar)

mse = nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=5e-6, momentum=0.9)

def train(model, optimizer, beta):
    size = len(X_dataloader.dataset)
    model.train()
    for batch, X in enumerate(X_dataloader):
        X = X[0].to(device)

        (mean, logvar, img) = model(X)
        loss_mse = mse(img, X)
        loss_kl = kl_std(mean, logvar)
        loss = loss_mse + beta * loss_kl

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

def test(model):
    size = len(X_dataloader.dataset)
    num_batches = len(X_dataloader)
    model.eval()
    test_loss_mse, test_loss_kl, correct = 0, 0, 0
    with torch.no_grad():
        for X in X_dataloader:
            X = X[0].to(device)
            (mean, logvar, img) = model(X)
            test_loss_mse += mse(img, X).item()
            test_loss_kl += kl_std(mean, logvar).item()
    test_loss_mse /= num_batches
    test_loss_kl /= num_batches
    correct /= size
    print(f"Test Error: \n Avg MSE loss: {test_loss_mse:>8f} \n Avg KL loss: {test_loss_kl:>8f} \n")

epochs = 800
ncycle = 2 # see "Cyclical Annealing Schedule: A Simple Approach to Mitigating KL Vanishing"
def beta_of(t):
    n = epochs / (2 * ncycle)
    i = t / n
    if i % 2 == 0:
        return i / n
    else:
        return 1

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(model, optimizer, beta_of(t))
    test(model)
print("Done!")

def show(rgb):
    img = np.empty([32, 32, 3])
    for i in range(32):
        for j in range(32):
            for k in range(3):
                img[i, j, k] = rgb[k, i, j]
    plt.axis('off')
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img)

def gen(code):
    with torch.no_grad():
        show(model.decode(code).numpy()[0])

def compare(i):
    show(X_dataset[i][0])
    gen(model.encode(torch.unsqueeze(X_dataset[i][0], 0)))

def fake():
    gen(torch.randn([1, lcode]))

for i in range(8*8):
    plt.subplot(8, 8, i+1)
    fake()
plt.subplots_adjust(wspace=0, hspace=0)
plt.show()

for i in range(32):
    t = random.randint(a=0, b=len(X_train) - 1)
    plt.subplot(8, 8, 2 * i + 1)
    show(X_dataset[t][0])
    plt.subplot(8, 8, 2 * i + 2)
    gen(model.encode(torch.unsqueeze(X_dataset[t][0], 0)))
plt.subplots_adjust(wspace=0, hspace=0)
plt.show()

for i in range(8):
    t1 = random.randint(a=0, b=len(X_train) - 1)
    t2 = random.randint(a=0, b=len(X_train) - 1)
    c1 = model.encode(torch.unsqueeze(X_dataset[t1][0], 0))
    c2 = model.encode(torch.unsqueeze(X_dataset[t2][0], 0))
    for a in range(11):
        plt.subplot(8, 11, i * 11 + a + 1)
        a /= 10
        gen(a * c1 + (1 - a) * c2)
plt.subplots_adjust(wspace=0, hspace=0)
plt.show()
