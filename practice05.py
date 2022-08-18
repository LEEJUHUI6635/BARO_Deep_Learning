import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

df_raw = pd.read_csv('./BostonHousing.csv')
# print(df_raw)
# print(df_raw.shape) #[506, 14]
# print(type(df_raw)) #frame.DataFrame

#Data preprocessing
x_raw = df_raw.drop(['medv'], axis = 1)
# print(x_raw)

y_raw = df_raw['medv']
# print(y_raw)

#Train dataset and Test dataset split
x_train, x_test, y_train, y_test = train_test_split(x_raw, y_raw, test_size = 0.3, random_state = 1234)

# print(x_train.shape) #[354, 13]
# print(x_test.shape) #[152, 13]

# print(y_train.shape) #[354, ]
# print(y_test.shape) #[152, ]

#Standardization -> x dataset에 대해서 정규화, test 시에도 정규화를 하고 넣어줘야 한다.
scaler = StandardScaler() #객체 생성
scaler.fit(x_train)
x_train_scale = scaler.transform(x_train)
x_test_scale = scaler.transform(x_test)

# print(x_train_scale)
# print(type(x_train_scale)) #ndarray
# print(type(y_train)) #series.Series

#Tensor화
x_train_tensor = torch.FloatTensor(x_train_scale)
x_test_tensor = torch.FloatTensor(x_test_scale)

y_train_tensor = torch.FloatTensor(y_train.values)
y_test_tensor = torch.FloatTensor(y_test.values)

# print(x_train_tensor.shape) #[354, 13]
# print(type(x_train_tensor)) #Tensor

#Hyper parameter
batch_size = 354
input_size = 13
hidden_size = 32
hidden2_size = 16
output_size = 1
learning_rate = 0.001
nb_epochs = 700

#Batch 학습을 위한 Dataloader, Dataset 
#Dataset을 만들어주는 이유는 Dataloader에 넣어줘야 하기 때문
train_data = data_utils.TensorDataset(x_train_tensor, y_train_tensor)
train_dataloader = data_utils.DataLoader(dataset = train_data, batch_size = batch_size, shuffle = True, drop_last = True)

#model 설정
class MLPModel(nn.Module):
    def __init__(self, input_size, hidden_size, hidden2_size, output_size):
        super(MLPModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.hidden2_size = hidden2_size
        self.output_size = output_size
        self.Linear1 = nn.Linear(self.input_size, self.hidden_size)
        self.Linear2 = nn.Linear(self.hidden_size, self.hidden2_size)
        self.Linear3 = nn.Linear(self.hidden2_size, self.output_size)
        self.ReLU = nn.ReLU()
        
    def forward(self, x):
        out = self.Linear1(x)
        out = self.ReLU(out) #image 복원의 문제에서는 ReLU를 거의 사용하지 않는다.
        out = self.Linear2(out)
        out = self.ReLU(out)
        out = self.Linear3(out)
        return out #Regression이기 때문에, 마지막 layer에서는 activation을 하지 않는다.

model = MLPModel(input_size, hidden_size, hidden2_size, output_size)

#optimizer, criterion 설정
#Regression -> criterion : MSELoss
optimizer = optim.SGD(model.parameters(), lr = learning_rate)
criterion = nn.MSELoss()

#train
train_loss_list = []
test_loss_list = []

for epoch in range(nb_epochs):
    #Batch 학습을 위해
    for idx, [X, Y] in enumerate(train_dataloader):
        model.train()
        y_predict = model(X)
        # print(X.shape) #[128, 13]
        # print(y_predict.shape) #[128, 1]
        # print(Y.shape) #[128]
        y_predict = y_predict.squeeze()
        # print(y_predict.shape)
        train_loss = criterion(y_predict, Y)
        
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        
        model.eval()
        y_test_predict = model(x_test_tensor)
        y_test_predict = y_test_predict.squeeze()
        test_loss = criterion(y_test_predict, y_test_tensor)
    
    #한 epoch 마다, 
    train_loss_list.append(train_loss.item())
    test_loss_list.append(test_loss.item())
    
    print('Epoch : {}, Train Loss : {}, Test Loss : {}'.format(epoch, train_loss_list[-1], test_loss_list[-1]))
    
#그래프 그리기
plt.figure() #객체 생성
plt.plot(train_loss_list)
plt.plot(test_loss_list)
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc = 'upper right')
plt.savefig('MLP_loss.png')
