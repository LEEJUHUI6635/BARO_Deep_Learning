import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

#Binary Classification -> sigmoid + Binary cross entropy

torch.manual_seed(1234)

df_raw = pd.read_csv('./BreastCancer.csv')
print(df_raw)
print(df_raw.shape) #[569, 33]
print(type(df_raw)) #DataFrame

#Dataset preprocessing
x_train = df_raw.drop(['id', 'Unnamed: 32'], axis = 1) #axis = 1 -> 한 열을 삭제한다.
print(x_train)
print(x_train.shape) #[569, 31]
print(type(x_train))

x_train = x_train.drop(['diagnosis'], axis = 1)
y_train = df_raw['diagnosis']
print(x_train.shape) #[569, 30] -> 30개의 features
print(y_train.shape) #[569, ]

print(type(y_train)) #series.Series
diag = {'M' : 1, 'B' : 0}
y_train = y_train.replace(diag)
print(y_train)

#Train dataset and Test dataset Split
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size = 0.2, random_state = 1234)
print(x_train.shape) #[455, 30]
print(x_test.shape) #[114, 30]

print(y_train.shape) #[455, ]
print(y_test.shape) #[114, ]

#Standardization -> x data에 대해서 정규화, test할 때에도 정규화해야 한다.
scaler = StandardScaler()
scaler.fit(x_train)
x_train_scale = scaler.transform(x_train)
x_test_scale = scaler.transform(x_test)

print(x_train_scale)
print(type(x_train_scale)) #ndarray
print(type(y_train)) #series.Series

x_train_tensor = torch.FloatTensor(x_train_scale)
x_test_tensor = torch.FloatTensor(x_test_scale)
y_train_tensor = torch.FloatTensor(y_train.values)
y_test_tensor = torch.FloatTensor(y_test.values)

#Hyper parameter
batch_size = 128
learning_rate = 0.01
input_size = 30
output_size = 1
nb_epochs = 200

#Batch학습을 위해 Dataloader
train_data = data_utils.TensorDataset(x_train_tensor, y_train_tensor)

train_dataloader = data_utils.DataLoader(dataset = train_data, batch_size = batch_size, shuffle = True, drop_last = True)

#model 설정 -> input_size = 30 features, output_size = 1
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(LogisticRegressionModel, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.Linear = nn.Linear(self.input_size, self.output_size) #Linear -> one layer = Perceptron
        self.Sigmoid = nn.Sigmoid()        
        
    def forward(self, x):
        out = self.Linear(x)
        out = self.Sigmoid(out) #0~1
        return out

model = LogisticRegressionModel(input_size, output_size)

#optimizer, criterion 설정
optimizer = optim.SGD(model.parameters(), lr = learning_rate)
criterion = nn.BCELoss() #0 or 1 -> Binary Cross Entropy

#train
train_loss_list = []
test_loss_list = []
train_accuracy_list = []
test_accuracy_list = []

#0과 1 중 하나의 정확한 정답이 있기 때문에, 정답을 맞췄을 때의 확률을 accuracy로 나타낼 수 있다.
for epoch in range(nb_epochs):
    #배치 학습
    for idx, [X, Y] in enumerate(train_dataloader):
        model.train()
        # print(X.shape) #[100, 30]
        # print(Y.shape) #[100]
        y_predict = model(X)
        # print(y_predict.shape)
        y_predict = y_predict.reshape(-1)
        # print(y_predict.shape) #[100]
        train_loss = criterion(y_predict, Y)
        
        #Accuracy 계산
        prediction = [1 if x > 0.5 else 0 for x in y_predict.data.numpy()]
        # print(prediction)
        # print(type(prediction)) #list
        # print(Y.shape)
        # print(type(Y.numpy())) #array
        train_accuracy = (prediction == Y.numpy()).sum() #batch_size의 data 각각을 비교하고 싶기 때문에, Y.numpy() -> 각각을 비교할 수 있다.
        # print(train_accuracy)
        # print(type(accuracy)) #numpy

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        
        model.eval() #Batch로 학습하지 않고, 한 번에 test
        y_test_predict = model(x_test_tensor)
        # print(x_test_tensor.shape) #[114, 30] 
        # print(y_test_predict.shape) #[114, 1]
        # print(y_test_tensor.shape) #[114]
        y_test_predict = y_test_predict.reshape(-1)
        # print(y_test_predict.shape) #[114]
        test_loss = criterion(y_test_predict, y_test_tensor)
        
        #Accuracy 계산
        test_prediction = [1 if x > 0.5 else 0 for x in y_test_predict.data.numpy()]
        test_accuracy = (test_prediction == y_test_tensor.numpy()).sum()
        # print(test_accuracy)
        # print(type(test_accuracy))
    
    train_accuracy = (train_accuracy * 100 / len(y_predict))
    test_accuracy = (test_accuracy * 100 / len(y_test_predict))
    # print(len(y_predict)) #[100]
    
    train_accuracy_list.append(train_accuracy)
    test_accuracy_list.append(test_accuracy)
        
    #한 epoch 끝날 때마다,
    train_loss_list.append(train_loss.item())
    test_loss_list.append(test_loss.item())
    
    print('Epoch : {}, Train Loss : {}, Test Loss : {}, Train Accuracy : {}, Test Accuracy : {}'.format(epoch + 1, train_loss_list[-1], test_loss_list[-1], train_accuracy_list[-1], test_accuracy_list[-1]))

#모델 파라미터 불러오기
print(model.state_dict())

#Accuracy 그래프 그리기
plt.figure() #객체 생성
plt.plot(train_accuracy_list)
plt.plot(test_accuracy_list)
plt.title('Model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc = 'upper left')
plt.savefig('train_test_accuracy.png')

#Loss 그래프 그리기
plt.figure() #객체 생성
plt.plot(train_loss_list)
plt.plot(test_loss_list)
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc = 'upper left')
plt.savefig('train_test_loss.png')