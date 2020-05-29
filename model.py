import torch
import torch.nn as nn
import torch.optim as optim
import math

import numpy as np

import matplotlib.pyplot as plt

class Model(nn.Module):
    def __init__(self, layer_size=126):
        super(Model, self).__init__()

        self.layer_size = layer_size

        self.f1 = nn.Linear(layer_size, 512)
        self.tanh1 = nn.Tanh()
        self.f2 = nn.Linear(512, 1024)
        self.tanh2 = nn.Tanh()
        self.f3 = nn.Linear(1024, 512)
        self.tanh3 = nn.Tanh()
        self.f4 = nn.Linear(512, 32)
        self.tanh4 = nn.Tanh()
        self.f5 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.f1(x)
        x = self.tanh1(x)
        x = self.f2(x)
        x = self.tanh2(x)
        x = self.f3(x)
        x = self.tanh3(x)
        x = self.f4(x)
        x = self.tanh4(x)
        x = self.f5(x)
        x = self.sigmoid(x)

        return x

class Model2(nn.Module):
    def __init__(self):
        super(Model2, self).__init__()

        layer_size = 126

        self.f1 = nn.Linear(layer_size, 500)
        self.tanh = nn.Tanh()
        self.f2 = nn.Linear(500, 50)
        self.prelu = nn.PReLU()
        self.f3 = nn.Linear(50, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.f1(x)
        x = self.tanh(x)
        x = self.f2(x)
        x = self.prelu(x)
        x = self.f3(x)
        x = self.sigmoid(x)

        return x

def one_hot(activate, total):
    to_ret = np.zeros(total)
    to_ret[activate] = 1
    return to_ret

def train(data, num_epochs = 50, batch_size = 5, print_every=500):
    m = Model() 
    # m = Model2() 

    opt = optim.Adam(m.parameters(), lr=0.005)
    criterion = nn.BCELoss()

    num_batches = int(math.ceil(float(data.shape[0]) / float(batch_size)))

    losses = []

    for epoch in range(num_epochs):

        p = np.random.permutation(len(data))
        data = data[p] # shuffle

        running_loss = 0.0

        for batch in range(num_batches):

            opt.zero_grad()

            d = torch.from_numpy(data[batch * batch_size:(batch+1) * batch_size, :])

            y,x = d[:,0],d[:,1:]

            y = nn.ReLU()(y - 0.01) + 0.005

            x = x.float()
            labels = y.float()

            output = m(x)

            # print('Expected {} but was {}'.format(y[0], output[0]))

            loss = criterion(output, labels)

            running_loss += loss.item()

            loss.backward()
            opt.step()

            if batch % print_every == print_every - 1:
                print('{}: {} / {} - {}'.format(epoch + 1, batch + 1, num_batches, running_loss / print_every))
                losses.append(running_loss / print_every)
                running_loss = 0.0

    torch.save(m.state_dict(), 'weights.pyt')

    return losses

if __name__ == "__main__":
    data = np.load('data.npy')
    print(data.shape)
    losses = train(data)
    plt.title('Loss vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(losses)
    plt.show()