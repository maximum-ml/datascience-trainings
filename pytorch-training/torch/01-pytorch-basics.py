# https://jovian.ai/aakashns/01-pytorch-basics

import torch

t1 = torch.tensor(4.)
print(f't1={t1}')
print(f't1.size={t1.size()}')
print(t1.dtype)


t2 = torch.tensor([1., 2, 3, 4])
print(f't2={t2}')
print(f't2.size={t2.size()}')

# Matrix
t3 = torch.tensor([[5., 6],
                   [7, 8],
                   [9, 10]])
print(f't3={t3}')
print(f't3.size={t3.size()}')

# 3D Array
t4 = torch.tensor([
    [[11, 12, 13],
     [13, 14, 15]],
    [[15, 16, 17],
     [17, 18, 19.]]])
print(f't4={t4}')
print(f't4.shape={t4.shape}')
print(f't4.size={t4.size()}')   # shape zwraca to samo co size()
# print(f't4.shape[0]={t4.size([])}')

# Create a tensor with a fixed value for every element
t5 = torch.full((3, 2), 42)
print(f't5={t5}')

# Concatenate two tensors with compatible shapes
t6 = torch.cat((t3, t5))
print(f't6={t6}')

# Change the shape of a tensor
t7 = t6.reshape(3, 2, 2)
print(f't7={t7}')

##########################################################

x = torch.tensor(3., requires_grad=True)
w = torch.tensor(4., requires_grad=True)
b = torch.tensor(5., requires_grad=True)

y = w * x + b
print(f'y={y}')
print(f'type(y)={type(y)}')

# Compute derivatives
yb = y.backward()
print(f'yb={yb}')

# Display gradients
print('dy/dx(x):', x.grad)
print('dy/dw(w):', w.grad)
print('dy/db(b):', b.grad)