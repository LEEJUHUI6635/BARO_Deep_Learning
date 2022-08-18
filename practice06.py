import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils

import torchvision.datasets as dsets
import torchvision.transforms as transforms

import matplotlib.pyplot as plt

#MNIST + FCN(MLP) -> image classification에 CNN이 아닌 FCN을 적용한 예시
#MNIST image dataset의 각각의 pixel이 처음 input의 각각의 node로 들어간다.
#Q. cross entropy function에 들어갈 때, 값의 제한이 있는가? 0~1 사이의 값이 들어가야 한다든지?
torch.manual_seed(1234)

#Transform
#Q. MNIST dataset은 mean과 std로 정규화하지 않아도 되는가?
#Q. ToTensor : [0, 255] -> [0.0, 1.0]이 이미 정규화를 시킨 것인가? -> 단순히 255로 나눠준 것
transform = transforms.Compose([
    transforms.ToTensor(), #ToTensor -> [0, 255] -> [0.0, 1.0]
    transforms.Normalize(mean = (0.5, ), std = (0.5, )) #흑백이기 때문 -> Standardization을 하면 더 성능이 좋아지는 듯
])

#Dataset 불러오기 -> MNIST dataset은 train dataset과 test dataset을 따로 불러올 수 있다.
train_data = dsets.MNIST(root = './MNISTdata', train = True, transform = transform, download = True)
test_data = dsets.MNIST(root = './MNISTdata', train = False, transform = transform, download = True)

#Hyper parameter
batch_size = 100
learning_rate = 0.1
input_size = 784 #image size = 28 x 28개의 pixel이 input으로 들어간다. 각각의 node로 들어가야 하기 때문에, 일렬로 펴서 넣어줘야 한다.
hidden_size = 256
output_size = 10 #0~9까지의 class
nb_epochs = 5

#Batch학습 -> Dataloader
train_dataloader = data_utils.DataLoader(dataset = train_data, batch_size = batch_size, shuffle = True, drop_last = True)
test_dataloader = data_utils.DataLoader(dataset = test_data, batch_size = batch_size, shuffle = False, drop_last = False)

#Q. MNIST dataset의 마지막 layer에서 10개의 node들 중에서 가장 큰 값을 구하는 것인가?

#model 설정
class MNISTClassifierModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MNISTClassifierModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.Linear1 = nn.Linear(self.input_size, self.hidden_size)
        self.Linear2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.Linear3 = nn.Linear(self.hidden_size, self.output_size)
        self.ReLU = nn.ReLU()
        
    def forward(self, x): 
        out = self.Linear1(x) #처음에 input image의 784의 pixel이 Linear1을 통과한다.
        out = self.ReLU(out) #image classification -> ReLU -> 흑백, 음의 값들은 날린다.
        out = self.Linear2(out)
        out = self.ReLU(out) #ReLU와 같은 활성화 함수는 비선형성을 더해준다.
        out = self.Linear3(out) #사실 마지막 layer에 ReLU가 들어가도 될 것 같은데, 마지막 layer라 굳이 넣지 않은 것 같다.
        return out

model = MNISTClassifierModel(input_size, hidden_size, output_size)

#optimizer, criterion 설정
optimizer = optim.SGD(model.parameters(), lr = learning_rate)
criterion = nn.CrossEntropyLoss() #softmax가 포함되어 있는 것 같다. 따라서 값이 나오면, 가장 확률이 큰 값의 index를 반환해주는 것인가?
#Loss function도 구현해봐야 할 듯?

#train
train_loss_list = []

for epoch in range(nb_epochs):
    #배치 학습
    for idx, [X, Y] in enumerate(train_dataloader):
        # print(X.shape) #[128, 1, 28, 28] -> 한 줄로 만든다.
        X = X.reshape(-1, 784) 
        # print(X.shape) #[128, 784]
        y_predict = model(X) 
        # print(y_predict.shape) #[128, 10] -> label -> Q. CrossEntropyLoss 뜯어봐야 할 듯 !
        # print(Y.shape) #[128]
        train_loss = criterion(y_predict, Y)
        
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

    #한 epoch 마다
    train_loss_list.append(train_loss.item())
    
    print('Epoch : {}, Train Loss : {}'.format(epoch + 1, train_loss_list[-1]))

#train loss 그래프 그리기
plt.figure() #객체 생성
plt.plot(train_loss_list, label = 'train')
plt.legend(loc = 'upper left')
plt.savefig('MNIST_loss.png')

#Test
with torch.no_grad(): #
    correct = 0
    for idx, [X, Y] in enumerate(test_dataloader):
        X = X.reshape(-1, 784)
        y_predict = model(X)
        # print(y_predict.shape) #[100, 10]
        #y_predict의 10에 대해서 가장 큰 값의 index를 구해야 한다. -> softmax + argmax(Cross Entropy Loss)
        #softmax는 입력받은 값을 출력으로 0~1 사이의 값으로 모두 정규화하며 출력 값들의 총합은 항상 1이 되는 특성을 가진 함수이다.
        y_predict_softmax = F.softmax(y_predict, dim = 1) 
        # print(y_predict_softmax)
        # print(y_predict_softmax.shape) #[100, 10]
        # print(type(y_predict_softmax)) #Tensor
        #softmax로 모든 값들을 0~1로 정규화하였으니, argmax를 통해서 확률이 가장 큰 값을 추출해야 한다.
        y_predict_argmax = torch.argmax(y_predict_softmax, dim = 1)
        # print(y_predict_argmax)
        # print(y_predict_argmax.shape) #[100]
        # print(Y.shape) #[100]
        correct += (y_predict_argmax == Y).sum().item() #item() -> Tensor -> int
        # print(correct) 
        # print(type(correct)) #int
   
    print(len(test_dataloader)) #100
    accuracy = 100 * correct / (batch_size * len(test_dataloader))
    print('Accuracy of test images : {}'.format(accuracy))