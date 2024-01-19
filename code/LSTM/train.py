import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable
from tqdm import tqdm_notebook
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from model import LSTM

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

data = pd.read_csv('../dataset/chap07/data/SBUX.csv')

# 형식 변환
data['Date'] = pd.to_datetime(data['Date'])
# Date를 행으로 만들어버림. 이제 행의 인덱스는 0부터 n-1까지가 아니라 Date의 값들이 됨
data.set_index('Date', inplace=True)
data['Volume'] = data['Volume'].astype(float)
x = data.iloc[:,:-1]
y = data.iloc[:, 5:6]

#값을 0과1사이로 조절한다. 같은 열에 대해서만!! 
ms = MinMaxScaler()

#값을 평균이 0 표준편차가 1인 데이터로 정규화 한다. 같은 열에 대해서만!!
ss = StandardScaler()
X_ss = ss.fit_transform(x)
y_ms = ms.fit_transform(y)

X_train = X_ss[:200, :]
X_test = X_ss[200:, :]

y_train = y_ms[:200, :]
y_test = y_ms[200:, :]

X_train_tensors = Variable(torch.Tensor(X_train))
X_test_tensors = Variable(torch.Tensor(X_test))

y_train_tensors = Variable(torch.Tensor(y_train))
y_test_tensors = Variable(torch.Tensor(y_test))
# 왜 이렇게 했는지는 이따 보자고
X_train_tensors_f = torch.reshape(X_train_tensors, (X_train_tensors.shape[0], 1, X_train_tensors.shape[1]))
X_test_tensors_f = torch.reshape(X_test_tensors, (X_test_tensors.shape[0], 1, X_test_tensors.shape[1]))

# 변수값 설정
num_epochs = 10000
lr = 0.0001

input_size=5
hidden_size=2
num_layers=1

num_classes=1
# 한 시퀀스 당 하나의 줄만 넣을 것임.(여기선 하나의 줄이 1x5행렬로 구성됨)
model = LSTM(num_classes, input_size, hidden_size, num_layers, X_train_tensors_f.shape[1])

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

for epoch in range(num_epochs) :
    outputs = model.forward(X_train_tensors_f)
    optimizer.zero_grad()
    loss = criterion(outputs, y_train_tensors)
    loss.backward()
    
    optimizer.step()
    if epoch % 100 == 0 :
        print(f'Epoch : {epoch}, loss : {loss.item():1.5f}')



df_x_ss = ss.transform(data.iloc[:, :-1])
df_y_ms = ms.transform(data.iloc[:, -1:])

df_x_ss = Variable(torch.Tensor(df_x_ss))
df_y_ms = Variable(torch.Tensor(df_y_ms))
df_x_ss = torch.reshape(df_x_ss, (df_x_ss.shape[0], 1, df_x_ss.shape[1]))

train_predict = model(df_x_ss)
predicted = train_predict.data.numpy()
label_y = df_y_ms.data.numpy()

predicted = ms.inverse_transform(predicted)
label_y = ms.inverse_transform(label_y)
plt.figure(figsize=(10, 6))
plt.axvline(x=200, c='r', linestyle='--')

plt.plot(label_y, label='Actual Data')
plt.plot(predicted, label='Predicted Data')
plt.title('Time-Series Prediction')
plt.legend()
plt.show()





