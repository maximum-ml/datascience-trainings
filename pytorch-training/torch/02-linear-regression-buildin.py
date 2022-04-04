# https://jovian.ai/aakashns/02-linear-regression

import numpy as np
import torch


# Input (temp, rainfall, humidity)
np_inputs = np.array([[73, 67, 43],
                   [91, 88, 64],
                   [87, 134, 58],
                   [102, 43, 37],
                   [69, 96, 70],
                   [74, 66, 43],
                   [91, 87, 65],
                   [88, 134, 59],
                   [101, 44, 37],
                   [68, 96, 71],
                   [73, 66, 44],
                   [92, 87, 64],
                   [87, 135, 57],
                   [103, 43, 36],
                   [68, 97, 70]],
                  dtype='float32')

# Targets (apples, oranges)
np_targets = np.array([[56, 70],
                    [81, 101],
                    [119, 133],
                    [22, 37],
                    [103, 119],
                    [57, 69],
                    [80, 102],
                    [118, 132],
                    [21, 38],
                    [104, 118],
                    [57, 69],
                    [82, 100],
                    [118, 134],
                    [20, 38],
                    [102, 120]],
                   dtype='float32')

inputs = torch.from_numpy(np_inputs)
targets = torch.from_numpy(np_targets)

# print(inputs[:3])

from torch.utils.data import TensorDataset

# Define dataset
train_ds = TensorDataset(inputs, targets)

# print(f'train_ds:\n{train_ds[:3]}')
# train_ds[0:3]

from torch.utils.data import DataLoader
# Define data loader
batch_size = 5
train_dl = DataLoader(train_ds, batch_size, shuffle=True) # na początku losuje wiersze, dla False - bierze 5 pierwszych

print(f'train_dl.dataset: \n{train_dl.dataset[:]}')

for xb, yb in train_dl:
    print(xb)
    print(yb)
    break


#################
import torch.nn as nn
import torch.nn.functional as F

# Define model
print("MODEL")
model = nn.Linear(3, 2)
print(model.weight)
print(model.bias)

# Parameters
print("MODEL.PRAMS")
print(list(model.parameters()))

# Define loss function
loss_fn = F.mse_loss

loss = loss_fn(model(inputs), targets)
print(f'loss={loss}')


# Generate predictions
preds = model(inputs)
print(f'preds={preds}')

# Define optimizer
opt = torch.optim.SGD(model.parameters(), lr=1e-4)  # SGD - stochastic gradient decent
print(f'opt={opt}')


# Utility function to train the model
def fit(num_epochs, model, loss_fn, opt, train_dl):
    # Repeat for given number of epochs
    for epoch in range(num_epochs):

        # Train with batches of data
        for xb, yb in train_dl:
            # 1. Generate predictions
            pred = model(xb)

            # 2. Calculate loss
            loss = loss_fn(pred, yb)

            # 3. Compute gradients
            loss.backward()

            # 4. Update parameters using gradients (poprzez optimizera)
            opt.step()

            # 5. Reset the gradients to zero
            opt.zero_grad()

        # Print the progress
        if (epoch + 1) % 5 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss))

fit(100, model, loss_fn, opt, train_dl)

print(f'model: P={list(model.parameters())}, B={model.bias}')