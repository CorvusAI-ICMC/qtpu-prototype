import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.nn.functional as func
import torch.nn as nn
from torch import optim
import numpy as np
import pandas as pd

datasetSize = 1000
dataset = np.random.rand(datasetSize, 2)
dataset = np.rint(dataset)
new_dataset = np.zeros((datasetSize, 3))
new_dataset[:, :2] = dataset
new_dataset[:, 2] = (dataset[:, 0] != dataset[:, 1]) #XOR
dataset = new_dataset


tensorDataset = torch.tensor(new_dataset, dtype=torch.float32)


inputs = tensorDataset[:, :2]
outputs = tensorDataset[:, 2]

torchDataset = TensorDataset(inputs, outputs)

trainSize = int(0.8 * len(torchDataset))
valSize = len(torchDataset) - trainSize
trainDataset, valDataset = random_split(torchDataset, [trainSize, valSize])


batchSize = 100
trainLoader = DataLoader(trainDataset, batch_size=batchSize, shuffle=True)
valLoader = DataLoader(valDataset, batch_size=batchSize, shuffle=True)

class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.lay1 = nn.Linear(2, 4)
        self.lay2 = nn.Linear(4, 2)
        self.lay3 = nn.Linear(2, 1)
    
    def forward(self, input):
        out = func.relu(self.lay1(input))
        out = func.relu(self.lay2(out))
        out = self.lay3(out)
        return out
    
model = NN()

optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.BCEWithLogitsLoss()

numEpochs = 50

for epoch in range(numEpochs):
    model.train()

    for batchInput, batchOutput in trainLoader:
        optimizer.zero_grad()
        outputs = model.forward(batchInput).squeeze()
        trainLoss = criterion(outputs, batchOutput)
        trainLoss.backward()
        optimizer.step()

    model.eval()
    valLoss = 0.0
    correct = 0 
    total = 0

    with torch.no_grad():
        for batchInput, batchOutput in valLoader:
            output = model.forward(batchInput).squeeze()
            predictions = torch.round(torch.sigmoid(output))
            correct += (predictions == batchOutput).sum().item()
            total += batchOutput.size(0)
            loss = criterion(output, batchOutput)
            valLoss += loss
    acc = correct / total
    print(f"Epoch {epoch}| train loss: {trainLoss / len(trainLoader):.4f} | val loss: {valLoss / len(valLoader):.4f} | acc: {acc}")









