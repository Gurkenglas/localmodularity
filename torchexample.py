import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
from tqdm import tqdm
import varname
import functools
import os

device='cpu'
#device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train(model, epochs=9, batchsize=0):
    dataset = datasets.MNIST #datasets.FashionMNIST
    train_dataloader,test_dataloader = (
        DataLoader(dataset(root="data",train=b,download=True,transform=ToTensor()),batch_size=batchsize) for b in (True, False))
    optimizer = torch.optim.Adam(model.parameters())
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        for X, y in tqdm(train_dataloader):
            pred = model(X.to(device))
            loss = nn.CrossEntropyLoss()(pred, y.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in tqdm(test_dataloader):
                pred = model(X.to(device))
                test_loss += nn.CrossEntropyLoss()(pred, y.to(device)).item()
                correct += (pred.argmax(1) == y.to(device)).type(torch.float).sum().item()
        print(f"Test Error: \n Accuracy: {correct/test_dataloader.__len__()/batchsize}, Avg loss: {test_loss/test_dataloader.__len__():>8f} \n")
    return model

def get_network():
    print(f'Using {device} device')
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28*28, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 10)
        ).to(device)
    weights = diskcache(lambda: train(model, epochs=9, batchsize=1000).state_dict())()
    model.load_state_dict(weights)
    return model

def diskcache(f):
    "Doesn't check args, only caches tensors."
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        name = varname.varname()
        filepath = name+'.pt'
        if os.path.exists(filepath): return torch.load(filepath)
        result = f(*args, **kwargs)
        torch.save(result, filepath)
        return result
    return wrapper