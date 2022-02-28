import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
from tqdm import tqdm

device='cpu'
#device = 'cuda' if torch.cuda.is_available() else 'cpu'
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

NeuralNetwork = lambda: nn.Sequential(
    nn.Flatten(),
    nn.Linear(28*28, 512),
    nn.ReLU(),
    nn.Linear(512, 512),
    nn.ReLU(),
    nn.Linear(512, 10),
)

def train_loop(dataloader, model, loss_fn, optimizer):
    for batch, (X, y) in enumerate(tqdm(dataloader)):
        pred = model(X.to(device))
        loss = loss_fn(pred, y.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)


def test_loop(dataloader, model, loss_fn):
    num_batches = dataloader.__len__()
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in tqdm(dataloader):
            pred = model(X.to(device))
            test_loss += loss_fn(pred, y.to(device)).item()
            correct += (pred.argmax(1) == y.to(device)).type(torch.float).sum().item()

    test_loss /= num_batches
    print(f"Test Error: \n Correct: {correct}, Avg loss: {test_loss:>8f} \n")

def train_network(dataset="MNIST"):
    learning_rate = 1e-3
    batch_size = 64
    epochs = 5

    train_dataloader,test_dataloader = (
        DataLoader((datasets.MNIST if dataset == "MNIST" else datasets.FashionMNIST)
            (root="data",train=b,download=True,transform=ToTensor()),batch_size=batch_size) for b in (True, False)
        )
    
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