import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split #train dataset과 test dataset split
from sklearn.preprocessing import StandardScaler

#csv 파일을 가져와서 Linear Regression 수행
df_raw = pd.read_csv('./BostonHousing.csv')

print(df_raw)
print(df_raw.shape) #[506, 14] -> 506개의 data samples, label 14개
print(type(df_raw)) #DataFrame

print(df_raw.head()) #상위 5개까지 확인

#Dataset Preprocessing
#medv 열은 result
x_raw = df_raw.drop(['medv'], axis = 1) #axis = 1 -> 열에 대해서 삭제
y_raw = df_raw['medv']
print(x_raw) #[506, 13]
print(y_raw)

#Train data and Test data split
x_train, x_test, y_train, y_test = train_test_split(x_raw, y_raw, test_size = 0.3, random_state = 1234)
print(x_train.shape) #[354, 13]
print(x_test.shape) #[152, 13]

print(y_train.shape) #[354, ]
print(y_test.shape) #[152, ]

print(type(y_train)) #pandas.core.series.Series

#Standardization -> x dataset은 서로 mean과 variance가 매우 다르기 때문에 이를 조절해야, 빠르게 수렴할 수 있다.
scaler = StandardScaler() #객체 생성 -> 평균 0, 분산 1인 Gaussian distribution
scaler.fit(x_train) #x_train에 fit한 mean과 variance를 찾는다.
x_train_scale = scaler.transform(x_train)
x_test_scale = scaler.transform(x_test) #test 시에도 standardization을 수행해야 한다.
print(x_train_scale)
print(x_test_scale)

print(type(x_train_scale)) #ndarray

#Tensor화
x_train_tensor = torch.FloatTensor(x_train_scale)
y_train_tensor = torch.FloatTensor(y_train.values) #CSV 파일 -> Tensor화 : values

x_test_tensor = torch.FloatTensor(x_test_scale)
y_test_tensor = torch.FloatTensor(y_test.values)

#Hyper parameter
batch_size = 100
nb_epochs = 200
input_size = 13
output_size = 1
learning_rate = 0.1

#Batch 학습 -> data loader 생성 -> Supervised learning이기 때문에, label - data로 묶어서 하나의 dataset으로 만들어야 한다.
train_data = data_utils.TensorDataset(x_train_tensor, y_train_tensor)
print(train_data)
print(type(train_data)) #TensorDataset
train_dataloader = data_utils.DataLoader(dataset = train_data, batch_size = batch_size, shuffle = True, drop_last = True) #drop_last = True -> 나머지는 버리겠다.

#model 설정 -> Linear Regression, input_size = 13(features), output_size = 1
model = nn.Linear(input_size, output_size)
#input_size의 feature는 결국 input layer의 node를 의미한다.
#output_size의 feature 1개는 결국 output layer의 node를 의미한다. 
#Linear Model, FCN(MLP)의 feature는 CNN의 feature(channel)의 의미와 다르게 node를 나타낸다.
#Linear Model, FCN(MLP)는 3차원이 아닌 2차원

#optimizer, criterion 설정
optimizer = optim.SGD(model.parameters(), lr = learning_rate)
criterion = nn.MSELoss() #Linear regression -> MSE loss를 사용한다.

#train
train_loss_list = []
test_loss_list = []

for epoch in range(nb_epochs):
    #배치학습을 위해
    for idx, [X, Y] in enumerate(train_dataloader):
        model.train() #학습할 때와 추론할 때 다르게 동작하는 Layer들을 Training mode로 바꾸어 준다.
        #예를 들어, Batch normalization은 학습할 때 Batch Statistics를 이용한다. Dropout Layer는 주어진 확률에 따라 활성화된다.
        # print(X.shape) #[100, 13] : batch_size x 13
        y_predict = model(X)
        # print(y_predict.shape) #[100, 1] : batch_size x 1
        # print(Y.shape) #[100]
        y_predict = y_predict.reshape(-1) 
        # print(y_predict.shape)
        train_loss = criterion(y_predict, Y)
        
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        
        model.eval() #eval mode는 batch로 test할 필요는 없다. 한 번에 test loss 계산
        #Q.eval mode이면, test dataloader를 쓸 수 있는 것인가?
        #Batch normalization은 학습할 때 사용된 Batch Statistics를 통해 결정된 Running Statistics를 이용하고, Dropout layer는 비활성화된다.
        y_test_predict = model(x_test_tensor)
        # print(y_test_predict.shape) #[152, 1]
        # print(y_test_tensor.shape) #[152]
        y_test_predict = y_test_predict.reshape(-1)
        test_loss = criterion(y_test_predict, y_test_tensor)
    #한 epoch가 끝나면, 
    train_loss_list.append(train_loss.item())
    test_loss_list.append(test_loss.item())
    
    print('Epoch : {}, Train Loss : {}, Test Loss : {}'.format(epoch, train_loss_list[-1], test_loss_list[-1]))
    
#Loss 그래프 그리기
plt.figure() #객체 생성
plt.plot(train_loss_list, label = 'train')
plt.plot(test_loss_list, label = 'test_loss')
plt.title('Model loss')
plt.legend(loc = 'upper right')
plt.savefig('loss_graph.png')