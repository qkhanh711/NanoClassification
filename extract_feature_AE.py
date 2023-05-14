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

def extractAutoEncoder(X_train, y_train, X_test, out_features = 5, num_epochs = 40, learning_rate = 1e-3, batch_size = 64):
    in_features = X_train.shape[1]
    

    trainset = np_to_tensor(X_train).float()
    testset = np_to_tensor(X_test).float()

    model = Autoencoder(in_features=in_features, out_features=out_features)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

    trained_model = train(num_epochs=num_epochs, dataloader=trainloader, model=model, criterion=criterion,
                        optimizer=optimizer)

    encoded_out_train = trained_model.encoder(trainset)
    X_train_AE = encoded_out_train.cpu().detach().numpy()

    calculate_time(SVC(), X_train_AE, y_train)
    return X_train_AE
