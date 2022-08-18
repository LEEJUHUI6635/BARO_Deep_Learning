import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

#Linear Regression

torch.manual_seed(1234)

#dataset 
#x_train : input, y_train : output -> y_train = Wx_train + b
x_train = torch.FloatTensor([[3.3], [4.4], [5.5], [6.1], [6.3],
                             [4.8], [9.7], [6.2], [7.9], [2.7],
                             [7.2], [10.1], [5.3], [7.7], [3.1]])

y_train = torch.FloatTensor([[1.7], [1.9], [2.09], [2.1], [1.9],
                             [1.3], [3.3], [2.5], [2.5], [1.1],
                             [2.7], [3.4], [1.5], [2.4], [1.3]])

plt.figure() #객체 생성
plt.scatter(x_train, y_train) #산점도 그리기
plt.savefig("mygraph.png")

#Hyper parameter
input_size = 1
output_size = 1
learning_rate = 0.001
nb_epochs = 50

#Linear model
model = nn.Linear(input_size, output_size) #feature(input_size)는 1개, layer 1개
#input size의 x data 1개가 결국 feature 1개로 작동한다.

#criterion, optimizer
#criterion -> MSE loss, cross entropy loss 등의 loss function은 쓰지 않는다.
optimizer = optim.SGD(model.parameters(), lr = learning_rate)
criterion = nn.MSELoss()

#train
#Batch 학습이 이루어지지 않는다. 따라서, 모든 15개의 data를 가지고 weight와 bias의 update가 이루어진다.
for epoch in range(nb_epochs):
    y_predict = model(x_train)
    # print(type(y_predict))
    # print(y_predict.shape) #[15, 1]
    loss = criterion(y_predict, y_train)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 2 == 0:
        print('Epoch : {}, Loss : {}'.format(epoch + 1, loss.item()))

#Q. model의 parameter들은 이미 저장이 된 것인가?

y_predict = model(x_train).detach().numpy() #Tensor를 Numpy로 변환

#그래프 그리기
plt.figure() #객체 생성
plt.plot(x_train, y_train, 'ro', label = 'data') #prediction
plt.plot(x_train, y_predict, label = 'linear function') #label
plt.legend(loc = 'upper left') #범례 위치 설정
plt.savefig('train_result.png')

#모델 파라미터 확인
print(model.state_dict()) #weight = 0.3279, bias = -0.0045