# https://jovian.ai/aakashns/02-linear-regression

import numpy as np
import torch

# Input (temp, rainfall, humidity)
nd_inputs = np.array([[73, 67, 43],
                   [91, 88, 64],
                   [87, 134, 58],
                   [102, 43, 37],
                   [69, 96, 70]], dtype='float32')

print(f'inputs.shape={nd_inputs.shape}')

print(f'inputs.type={type(nd_inputs)}')

# Targets (apples, oranges)
nd_targets = np.array([[56, 70],
                    [81, 101],
                    [119, 133],
                    [22, 37],
                    [103, 119]], dtype='float32')

print(f'targets.type={type(nd_targets)}')


# Convert inputs and targets to tensors

inputs = torch.from_numpy(nd_inputs)
targets = torch.from_numpy(nd_targets)

print(f'inputs={inputs}')
print(f'targets={targets}')

# Weights and biases
W = torch.ones(3, 2, requires_grad=True)    # 2 kolumny, 3 wiersze - format odwrotny niż przy tworzeniu tablic

print(f'W={W}')
B = torch.zeros(2, requires_grad=True)

# Y = xW + B  mamy dwie kolumny bo od razu liczymy współczynniki dla jabłek i pomarańczy

# @ represents matrix multiplication in PyTorch, and the .t method returns the transpose of a tensor.
def model(x):
    return x @ W + B

y = model(inputs)

print(f'y={y}')


def mse(t1, t2):
    diff = t1 - t2
    result = torch.sum(diff * diff) / diff.numel()
    return  result


loss = mse(y, targets)

print(f'loss={loss}, {type(loss)}')

loss.backward()

print(f'w.grad={W.grad}')



step = 1e-4

def move_step(w , b) :
    with torch.no_grad() :
        w -= w.grad * step
        b -= b.grad * step
    return (w, b)

W, B = move_step(W, B)

print(f'W={W}')
print(f'B={B}')

W.grad.zero_()
B.grad.zero_()

y = model(inputs)

loss = mse(y, targets)
print(f'loss={loss}, {type(loss)}')
loss.backward()


W, B = move_step(W, B)

print(f'W={W}')
print(f'B={B}')

W.grad.zero_()
B.grad.zero_()

y = model(inputs)

loss = mse(y, targets)
print(f'loss={loss}, {type(loss)}')

print("XXXXXXXXXXXXXXXXXXXX")

def predict(w, b, x):
    return x @ w + b


def calc_loss(act, goal):
    diff = act - goal
    result = torch.sum(diff * diff) / diff.numel()
    return result


def single_cycle(x, y, n_max) :
    w = torch.ones(3, 2, requires_grad=True)  # 2 kolumny, 3 wiersze - format odwrotny niż przy tworzeniu tablic
    b = torch.zeros(2, requires_grad=True)

    last_lost = -1.

    for i in range(n_max) :
        predictions = predict(w, b, x)
        lost = mse(predictions, y)

        if last_lost != -1.0 and lost > last_lost:
            print(f'L_LOSS={last_lost}, N={i}')
            return (w, b)

        last_lost = lost
        lost.backward()
        w, b = move_step(w, b)
        w.grad.zero_()
        b.grad.zero_()

    print(f'L_LOSS={last_lost}, N={n_max}')

    return (w, b)



single_cycle(inputs, targets, 600)