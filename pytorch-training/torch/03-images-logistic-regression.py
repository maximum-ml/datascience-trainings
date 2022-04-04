import torch
import torchvision
import matplotlib.pyplot as plt
import pylab

from torchvision.datasets import MNIST

# Download training dataset
dataset = MNIST(root='data/', download=False)

print(dataset)

print(len(dataset))

test_dataset = MNIST(root='data/', train=False)
print(len(test_dataset))

print(f'dataset[0]={dataset[0]}')

image, label = dataset[1]
# plt.ion()
# plt.imshow(image, cmap='gray')
# pylab.show()
print('Label:', label)

import torchvision.transforms as transforms

transformer = transforms.ToTensor()

tensor = transformer(image)


print(f'tensor.shape={tensor.shape}')

# plt.imshow(tensor[0, :, :], cmap='gray')
# pylab.show()

# labels1 = dataset[0:100].label
# print(labels1)

from torch.utils.data import random_split

# for i in range(1, 100):
#     print(f'L={dataset[i][0]}')
#     print(f'I={dataset[i][1]}')


train_ds, val_ds = random_split(dataset, [50000, 10000])
print(f'train_ds.len={len(train_ds)}, val_ds.len={len(val_ds)}')

# xxx = val_ds.data[0:100]

print(f'elements={val_ds[0]}')

from torch.utils.data import DataLoader

batch_size = 128

train_loader : DataLoader = DataLoader(train_ds, batch_size, shuffle=True)
val_loader : DataLoader = DataLoader(val_ds, batch_size)

input_size = 28*28
num_classes = 10

# [21] Logistic regression model
import torch.nn as nn
model = nn.Linear(input_size, num_classes)

print(f'model.weight.shape={model.weight.shape}')
print(f'model.bias.shape={model.bias.shape}')

