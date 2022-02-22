import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')


training_data = datasets.MNIST( #.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.MNIST(  #.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)



def preprocess(x, y):
    return x.to(device), y.to(device)

class WrappedDataLoader:
    def __init__(self, dl, func):
        self.dl = dl
        self.func = func

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        batches = iter(self.dl)
        for b in batches:
            yield (self.func(*b))

    def __size__(self):
        return len(self.dl.dataset)



class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits



def train_loop(dataloader, model, loss_fn, optimizer):
    size = dataloader.__size__()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = dataloader.__size__()
    num_batches = dataloader.__len__()
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

def train_network(dataset="MNIST"):

    learning_rate = 1e-3
    batch_size = 64
    epochs = 5

    if dataset=="MNIST":
        train_dataloader = WrappedDataLoader(DataLoader(datasets.MNIST(root="data",train=True,download=True,transform=ToTensor()), batch_size=batch_size), preprocess)
        test_dataloader = WrappedDataLoader(DataLoader(datasets.MNIST(root="data",train=False,download=True,transform=ToTensor()), batch_size=batch_size), preprocess)

    else:
        train_dataloader = WrappedDataLoader(DataLoader(datasets.FashionMNIST(root="data",train=True,download=True,transform=ToTensor()), batch_size=batch_size), preprocess)
        test_dataloader = WrappedDataLoader(DataLoader(datasets.FashionMNIST(root="data",train=False,download=True,transform=ToTensor()), batch_size=batch_size), preprocess)
    
    model = NeuralNetwork().to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    epochs = 10
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        test_loop(test_dataloader, model, loss_fn)
    print("Done!")

    return model