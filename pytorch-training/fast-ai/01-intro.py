import numpy as np
import torch as torch
import matplotlib.pyplot as plt
import math as math


data = [[1, 2, 3, ],
        [4, 5, 6]]

arr = np.array(data)
tns : torch.Tensor = torch.tensor(data)

print(f'arr={arr}')
print(f'tns={tns}')

sigmo = tns.sigmoid()
print(f'sigmo={sigmo}')

print("=============")

print(tns[:, 0])


def f(x): return x ** 2

def plot_function(func, start = -2.0, end = 2.0, step = 0.1):

    x = np.arange(start, end, step)
    y = [func(i) for i in x]
    # plotting the points
    plt.plot(x, y)

    # naming the x axis
    plt.xlabel('x')
    # naming the y axis
    plt.ylabel('y')

    # giving a title to my graph
    plt.title('f')

    # function to show the plot
    plt.show()


def plot_points(x, y, label):
    plt.plot(x, y, label=label)
    # naming the x axis
    # plt.xlabel('x')
    # naming the y axis
    # plt.ylabel('y')

    # giving a title to my graph
    plt.title(label)

    # function to show the plot


print("=============")

xt1 = torch.tensor([3.]).requires_grad_() # są dwie metody z udnerscore i bez - na czym polega różnica ?
yt = f(xt1)

yt.backward() # calculate gradients (it should be called calc_grad() - w tensorach z którch został policzony yt

print(f'yt.grad={xt1.grad}')
# xt2 = torch.tensor([3.])

# print(xt1.shape)
# print(xt2.shape)

time = torch.arange(-20, 20).float()

def f_speed(x): return 0.75 * (x - 9.5) ** 2 + 1 + torch.randn(1) * 5
# speed = 0.75 * (time - 9.5) ** 2 + 1 + torch.randn(20) * 3

y = f_speed(time)

def est_func(x, params):
    a, b, c = params
    return a * x ** 2 + b * x + c

def mse(preds, targets) : return torch.sqrt(((preds - targets)**2).mean())


# print(f'y={est_func(2, [3, 5, 7])}')

params = torch.randn(3).requires_grad_()
params_orig = params.clone()



# print(f'y_est={y_est.data}')

# loss = mse(y_est, y)
# print(f'loss={loss}, type={type(loss)}')
# loss.backward()






# plot_function(f_speed)

# r = torch.randn(10)
# print(f'r={r}')

LEARNING_RATE = 0.0002

print(f"PARAMS0={params}")


last_loss = 10000000.0
# loss = None

def iteration(x):
    est = est_func(time, params)
    # print(f"EST={est}")

    loss = mse(est, y)
    # print(f"[{x}] L={loss}, PARAMS={params}")
    # print(f"PARAMS={params}")
    loss.backward()
    # print(f"GRAD={params.grad.data}")
    params.data -= LEARNING_RATE * params.grad.data * math.sqrt(x)
    params.grad = None
    return loss

iter = 0


while True:
    iter += 1
    loss = iteration(iter)

    if last_loss <= loss.item():
        break
    else:
        last_loss = loss.item()

# while (loss == None or last_loss > loss.item):


# for i in range(5000):
#     iteration(i+1)

y_est = est_func(time, params)
y_est_orig = est_func(time, params_orig)


f_est_orig_0 = est_func(0.0, params_orig)
print(f"f_est_orig_0={f_est_orig_0}")

f_est_0 = est_func(0.0, params)
print(f"f_est_0={f_est_0}")



plot_points(time.numpy(), y_est.detach().numpy(), "y_est")
plot_points(time.numpy(), y_est_orig.detach().numpy(), "y_est_orig")
plot_points(time.numpy(), y.detach().numpy(), "y")
plt.legend()

plt.show()