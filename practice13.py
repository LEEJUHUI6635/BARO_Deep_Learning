import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils

import torchvision.datasets as dsets
import torchvision.transforms as transforms

import os

torch.manual_seed(1234)

device = ('cuda' if torch.cuda.is_available() else 'cpu')
    
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean = (0.5, ), std = (0.5, ))
])

train_data = dsets.MNIST(root = './MNISTdata', train = True, transform = transform, download = True)
test_data = dsets.MNIST(root = './MNISTdata', train = False, transform = transform, download = True)

#Hyper parameter
batch_size = 1000
learning_rate = 0.001
input_size = 1 #channel -> 흑백
hidden_size = 64 #channel
hidden2_size = 128 #channel
output_size = 10 #마지막 FCN layer에 대해서, output node의 개수
nb_epochs = 15
model_save_dir = './pre_trained/'

if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)
    
#Batch 학습을 위한 Dataloader
train_dataloader = data_utils.DataLoader(dataset = train_data, batch_size = batch_size, shuffle = True, drop_last = True)
test_dataloader = data_utils.DataLoader(dataset = test_data, batch_size = batch_size, shuffle = False, drop_last = False)

#model 설정
#input : batch_size x channel x image_size x image_size
#FCN에서는 2차원이기 때문에, 한 줄로 펴서 28 x 28의 픽셀을 넣어줬다면, CNN은 3차원이기 때문에, 4차원의 Tensor를 입력으로 넣어준다.
#input -> batch_size x 1 x 28 x 28
class CNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, hidden2_size, output_size, batch_size):
        super(CNNModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.hidden2_size = hidden2_size
        self.output_size = output_size
        self.batch_size = batch_size
        layers = []
        layers.append(nn.Conv2d(in_channels = self.input_size, out_channels = self.hidden_size, kernel_size = 1, stride = 1, padding = 1)) #out = (in + 2 * 1 - 1) / 1 + 1
        layers.append(nn.BatchNorm2d(num_features = self.hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.MaxPool2d(2)) #size가 1/2배로 줄어든다.
        layers.append(nn.Conv2d(in_channels = self.hidden_size, out_channels = self.hidden2_size, kernel_size = 5, stride = 2, padding = 0)) #out = (in + 2 * 0 - 5) / 2 + 1
        layers.append(nn.BatchNorm2d(num_features = self.hidden2_size))
        layers.append(nn.ReLU())
        layers.append(nn.MaxPool2d(2))
        
        self.cnn_layers = nn.Sequential(*layers)
        self.out = torch.randn(self.batch_size, 1, 28, 28)
        self.result = self.cnn_layers(self.out)
        self.out_reshape = self.result.reshape(self.batch_size, -1)
        self.out_channel = self.out_reshape.size(1)
        self.fc = nn.Linear(self.out_channel, 10)
        
    #Q. forward 안에 nn을 넣으면 연산이 안되는 것 같다?
    def forward(self, x):
        # #[batch_size, 1, 28, 28]
        # out = self.conv1(x) #(28+2-1)/1+1 = 30
        # #[batch_size, hidden_size, 30, 30]
        # out = self.batchnorm1(out)
        # out = self.relu(out)
        # out = self.maxpool(out) #[batch_size, hidden_size, 15, 15]
        
        # out = self.conv2(out) #(15-5)/2+1 = 6
        # #[batch_size, hidden2_size, 6, 6]
        # out = self.batchnorm2(out)
        # out = self.relu(out)
        # out = self.maxpool(out)
        # #[batch_size, hidden2_size, 3, 3]
        # out_reshape = out.reshape(batch_size, -1)
        #[batch_size, hidden2_size x 3 x 3]
        out = self.cnn_layers(x)
        out = out.reshape(batch_size, -1)
        out = self.fc(out)
        return out

model = CNNModel(input_size, hidden_size, hidden2_size, output_size, batch_size).to(device)

# input = torch.randn(128, 1, 28, 28)
# print(input.shape)
# output = model(input)
# print(output)
# print(output.shape)
# print(type(output))

#optimizer, criterion 설정
optimizer = optim.Adam(model.parameters(), lr = learning_rate) #betas = [0.9, 0.999] default
criterion = nn.CrossEntropyLoss()

#train
# train_loss_list = []
# for epoch in range(nb_epochs):
#     #배치 학습
#     for idx, [X, Y] in enumerate(train_dataloader):
#         X = X.to(device)
#         Y = Y.to(device)
#         y_predict = model(X)
    
#         train_loss = criterion(y_predict, Y)

#         optimizer.zero_grad()
#         train_loss.backward()
#         optimizer.step()

#     #한 epoch 마다,
#     train_loss_list.append(train_loss.item())
#     print('Epoch : {}, Train Loss : {}'.format(epoch + 1, train_loss_list[-1]))
    
#model parameter 저장
save_weight = os.path.join(model_save_dir, 'CNN_MNIST.pt')
# torch.save(model.state_dict(), save_weight)

#model parameter 불러오기
model.load_state_dict(torch.load(save_weight))

#Test
with torch.no_grad():
    #for Batch test
    accuracy = 0
    for idx, [X, Y] in enumerate(test_dataloader):
        X = X.to(device)
        Y = Y.to(device)
        # print(X.shape) #[1000, 1, 28, 28]
        # print(Y.shape) #[1000]
        y_predict = model(X)
        # print(y_predict)
        # print(y_predict.shape) #[1000, 10]
        #먼저 softmax 함수를 통해 모든 값을 0~1로 정규화해야 한다.
        y_predict_softmax = F.softmax(y_predict, dim = 1)
        # print(y_predict_softmax)
        # print(y_predict_softmax.shape) #[1000, 10]
        #argmax를 통해 값들 중 가장 큰 값의 index를 추출한다. 즉 classification을 수행한다.
        y_predict_argmax = torch.argmax(y_predict_softmax, dim = 1)
        # print(y_predict_argmax)
        correct = (y_predict_argmax == Y).sum()
        # print(correct)
        # print(correct.shape) #[]
        # print(type(correct)) #Tensor
        accuracy += correct
    length = len(test_dataloader) * batch_size
    accuracy = (accuracy * 100) / length
     
    print('CNN을 이용한 모델의 정확도 : {}%'.format(accuracy))