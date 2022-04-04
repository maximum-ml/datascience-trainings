import numpy as np
import torch as torch
import matplotlib.pyplot as plt
import math as math


# advanced tensor indexing

idx = range(5)
print(idx)

ten = torch.randn(5, 2)
print(ten)

choice = torch.tensor([1, 0, 0, 1, 1])
print(choice)

result = ten[idx, choice] # w pierszym wymiarze wybieramy po kolei a w drugim raz 1 raz 0

print(result)