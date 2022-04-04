import torch
# import jovian
import torchvision
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision.datasets.utils import download_url
from torch.utils.data import DataLoader, TensorDataset, random_split


DATASET_URL = "https://hub.jovian.ml/wp-content/uploads/2020/05/insurance.csv"
DATA_FILENAME = "./data/insurance.csv"
download_url(DATASET_URL, './data/')


#[5]
dataframe_raw = pd.read_csv(DATA_FILENAME)
print(dataframe_raw.head())

dataframe = dataframe_raw.copy(deep=True)

print(f'dataframe.size={dataframe.size}')
print(f'datafram2e.shape [rows]={dataframe.shape[0]}')
print(f'datafram2e.shape [cols]={dataframe.shape[1]}')

rand_str= 'Wojtek'

# Show all column names
print(dataframe.columns) # -> zwraca listę nazw kolumn

# show all column with data types:
for col_name in dataframe.columns:
    col_type = dataframe.dtypes[col_name]
    print(f'COL={col_name} - TYPE={col_type}')



input_cols = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']
output_cols = ['charges']
categorical_cols = ['sex', 'smoker', 'region']

def dataframe_to_arrays(dataframe):
    # Make a copy of the original dataframe
    dataframe1 = dataframe.copy(deep=True)
    # Convert non-numeric categorical columns to numbers
    for col in categorical_cols:
        dataframe1[col] = dataframe1[col].astype('category').cat.codes
    # Extract input & outupts as numpy arrays
    inputs_array = dataframe1[input_cols].to_numpy()
    targets_array = dataframe1[output_cols].to_numpy()
    return inputs_array, targets_array

inputs_array, targets_array = dataframe_to_arrays(dataframe)

print("INPUTS:")
print(inputs_array)
print(type(inputs_array))

print("TARGETS:")
print(targets_array)
print(type(targets_array))

inputs = torch.from_numpy(inputs_array).type(torch.float32)
targets = torch.from_numpy(targets_array).type(torch.float32)

print(inputs.dtype, targets.dtype)

dataset = TensorDataset(inputs, targets)

print("dataset:", dataset)


val_percent = 0.15 # between 0.1 and 0.2
num_rows = len(dataframe)
print(f'num_rows={num_rows}')
val_size = int(num_rows * val_percent)
train_size = num_rows - val_size

print(f'val_size={val_size}, train_size={train_size}')

train_ds, val_ds = random_split(dataset, [train_size, val_size])

batch_size = 100

train_loader = DataLoader(train_ds, batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size)

for xb, yb in train_loader:
    print("inputs:", xb)
    print("targets:", yb)
    break

# dataframe = dataframe.sample(int(0.95*len(dataframe)), random_state=int(ord(rand_str[0])))