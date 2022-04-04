import numpy as np
import torch as torch
import matplotlib.pyplot as plt
import math as math

import torchvision.transforms as transforms
from torchvision.datasets import MNIST

#------------ prepare data -----------------

test_dataset = MNIST(root='../data/', download=False, train=False)
train_dataset = MNIST(root='../data/', download=False, train=True)

print(f'test_ds.len={len(test_dataset)}')
print(f'train_ds.len={len(train_dataset)}')

# print(test_dataset[0])

# convert datasets into a map : label -> list of images
map = {}
for (image, label) in test_dataset:
    map.setdefault(label, []).append(image)

print(f'map.size={len(map)}')
print(f'map.keys={map.keys()}')

print(f'map.key[3].len={len(map[3])}')
print(f'map.key[7].len={len(map[7])}')

threes = map[3][0:10]
sevens = map[7][0:10]
print(f'threes.size={len(threes)}')
print(f'sevens.size={len(sevens)}')


transformer = transforms.ToTensor()

tensor = transformer(image)

#-------------------------------------------

def init_params(size, std = 1.0): return (torch.ran dn(size) * std).requires_grad_()

weights = init_params(28 * 28, 1.0)
bias = init_params(1, 1.0)

# print(f'weights={weights}')
# print(f'bias={bias}')