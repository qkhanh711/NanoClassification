from utils import *
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.model_selection import train_test_split

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Autoencoder(nn.Module):

    def __init__(self, in_features=None, out_features=None):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features, 100, bias=True),
            nn.ReLU(),
            nn.Linear(100, 50, bias=True),
            nn.ReLU(),
            nn.Linear(50, out_features, bias=True),
        )
        self.decoder = nn.Sequential(
            nn.Linear(out_features, 50, bias=True),
            nn.ReLU(),
            nn.Linear(50, 100, bias=True),
            nn.ReLU(),
            nn.Linear(100, in_features, bias=True),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
def np_to_tensor(np):
    return torch.from_numpy(np).float().to(device)

def train(num_epochs=None, dataloader=None, model=None, criterion=None, optimizer=None):
    for epoch in range(num_epochs):
        for data in dataloader:
            inputs = data
            inputs = Variable(inputs)
            output = model(inputs)
            loss = criterion(output, inputs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model
