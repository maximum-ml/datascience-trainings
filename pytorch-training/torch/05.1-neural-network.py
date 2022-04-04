import torch
import torchvision
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split

matplotlib.rcParams['figure.facecolor'] = '#ffffff'

dataset = MNIST(root='data2/', download=True, transform=ToTensor())     # to jest podklasa DataSet

print(f'dataset.type={type(dataset)}')

#[5]
image, label = dataset[0]
print(f'image.type={type(dataset)}, label.type={type(label)}')
print('image.shape:', image.shape)
# plt.imshow(image.permute(1, 2, 0), cmap='gray')

print('Label:', label)


val_size = 10000
train_size = len(dataset) - val_size

train_ds, val_ds = random_split(dataset, [train_size, val_size])
print(len(train_ds), len(val_ds))

print(f'train_ds.type={type(train_ds)}')

batch_size=128

train_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size*2, num_workers=0, pin_memory=True)

# [10]
# iter = 0;

# images, _ = train_loader.dataset[0:128]
# make_grid(images, nrow=16)
# plt.imshow(make_grid(images, nrow=16))

# to jest taki cwany sposób czytania obrazów z loadera - images to od razu paczka 128 obrazków - czyli tensor [128, 1, 28, 28]
# for images, _ in train_loader:
#     print(f'[{iter}]images.shape:', images.shape)
#     plt.figure(figsize=(16,8))
#     plt.axis('off')
#     plt.imshow(make_grid(images, nrow=16).permute((1, 2, 0)))
#     break

for images, labels in train_loader:
    print('images.shape:', images.shape)
    inputs = images.reshape(-1, 784)
    print('inputs.shape:', inputs.shape)
    break

input_size = inputs.shape[-1]
print('input_size:', input_size)
hidden_size = 32
print('outut_size:', hidden_size)

layer1 = nn.Linear(input_size, hidden_size) # 784 -> 32

#[15]
layer1_outputs = layer1(inputs)
print('layer1_outputs.shape:', layer1_outputs.shape)


layer1_outputs_direct = inputs @ layer1.weight.t() + layer1.bias
print('layer1_outputs_direct.shape:', layer1_outputs_direct.shape)

ret = torch.allclose(layer1_outputs, layer1_outputs_direct, 1e-3)
print(f'ret={ret}')

#[21]
relu_outputs = F.relu(layer1_outputs)

# jako sprawdzenie
print('min(layer1_outputs):', torch.min(layer1_outputs).item()) # item() zwraca element z tensora (to działa tylko dla tensora w którym jest tylko 1 elem)
print('min(relu_outputs):', torch.min(relu_outputs).item())



print(f'relu_outputs.shape={relu_outputs.shape}')

single_relu_output = relu_outputs[0,:]
print(f'single_relu_output.shape={single_relu_output.shape}')
print("single relu:\n", single_relu_output)


output_size = 10
layer2 = nn.Linear(hidden_size, output_size)

layer2_outputs = layer2(relu_outputs)
print(layer2_outputs.shape)

loss = F.cross_entropy(layer2_outputs, labels)

print(f'loss={loss}')

#[26]
# Expanded version of layer2(F.relu(layer1(inputs)))
outputs = (F.relu(inputs @ layer1.weight.t() + layer1.bias)) @ layer2.weight.t() + layer2.bias


plt.show()