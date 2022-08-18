import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils

import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torchvision.utils import save_image

import os

#Vanilla GAN -> FCN, MLP를 이용한다.

torch.manual_seed(1234)

device = ('cuda' if torch.cuda.is_available() else 'cpu')

save_dir = './GAN_model/save_weight/'
train_image_dir = './GAN_model/train_image/'
test_image_dir = './GAN_model/test_image/'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)
if not os.path.exists(train_image_dir):
    os.makedirs(train_image_dir)
if not os.path.exists(test_image_dir):
    os.makedirs(test_image_dir)
    
#Dataset 불러오기 -> Vanilla GAN은 test 시 image를 사용하는 것이 아니기 때문에, test dataset을 따로 만들지 않아도 된다.
transform = transforms.Compose([
    transforms.ToTensor(), #Q.0~1 or -1~1?
    transforms.Normalize(mean = (0.5, ), std = (0.5, )) #흑백
])

train_data = dsets.MNIST(root = './MNISTdata', train = True, transform = transform, download = True)

#Hyper parameter
batch_size = 128
learning_rate = 0.0002
noise_dim = 100
hidden_size = 256
hidden2_size = 512
hidden3_size = 1024
image_size = 28 * 28
nb_epochs = 100

#Batch 학습 -> DataLoader
train_dataloader = data_utils.DataLoader(dataset = train_data, batch_size = batch_size, shuffle = True, drop_last = True)

#Batch data 확인
for idx, [images, labels] in enumerate(train_dataloader):
    # print(images)
    break

#noise z 생성 -> FCN(2차원), noise 생성할 때 2차원에 맞춰서 생성하면 된다.
z = torch.randn(batch_size, noise_dim).to(device) #batch_size x noise dimension

#Generator 모델 생성
#input : noise z, output : batch_size x 1 x 28 x 28
#2차원인 FCN(MLP)으로 이루어진 GAN이기 때문에, input node는 noise의 dimension인 100, output node는 각 pixel을 생성해야 하기 때문에 784
#Generator에 dropout을 쓰지 않는 이유는 모든 feature가 중요하다고 생각하기 때문? image generation 문제에 있어서, 임의의 node를 제거하면 이미지 생성 방해
class GeneratorG(nn.Module):
    def __init__(self, noise_dim, hidden_size, hidden2_size, hidden3_size, image_size):
        super(GeneratorG, self).__init__()
        self.noise_dim = noise_dim
        self.hidden_size = hidden_size
        self.hidden2_size = hidden2_size
        self.hidden3_size = hidden3_size
        self.output_size = image_size
        layers = []
        layers.append(nn.Linear(self.noise_dim, self.hidden_size)) #image를 generation(reconstruction)하는 문제이기 때문에 ReLU보다는 leaky_relu 사용
        layers.append(nn.LeakyReLU(negative_slope = 0.2)) #leaky_relu(input, negative_slope = 0.01)
        layers.append(nn.Linear(self.hidden_size, self.hidden2_size))
        layers.append(nn.LeakyReLU(negative_slope = 0.2))
        layers.append(nn.Linear(self.hidden2_size, self.hidden3_size))
        layers.append(nn.LeakyReLU(negative_slope = 0.2))
        layers.append(nn.Linear(self.hidden3_size, self.output_size))
        layers.append(nn.Tanh()) #Tanh() : -1~1 -> train_dataloader에서 batch data를 확인해보면, -1~1로 정규화시킴을 볼 수 있다.
        
        self.layersG = nn.Sequential(*layers)
        
    def forward(self, x):
        out = self.layersG(x)
        
        return out
        
generatorG = GeneratorG(noise_dim, hidden_size, hidden2_size, hidden3_size, image_size).to(device)

# out = generatorG(z)
# print(out)
# print(out.shape) #[128, 784]
#이미지를 출력하기 위해, 위의 Tensor를 [batch_size, 1, 28, 28]로 변환하여 내보낸다.

#같은 noise로 여러 개의 fake image를 만들면, 서로 같은 fake image를 생성할까?
noise = torch.randn(batch_size, 100).to(device)
fake_image1 = generatorG(noise)
fake_image2 = generatorG(noise)
# print(fake_image1 == fake_image2) #True

#Discriminator 모델 생성
#input : image(batch_size x 784) -> output : classification -> loss function : Binary Cross Entropy
class DiscriminatorD(nn.Module): 
    def __init__(self, image_size, hidden3_size, hidden2_size, hidden_size):
        super(DiscriminatorD, self).__init__()
        self.input_size = image_size
        self.hidden3_size = hidden3_size
        self.hidden2_size = hidden2_size
        self.hidden_size = hidden_size
        layers = []
        layers.append(nn.Linear(self.input_size, self.hidden3_size))
        layers.append(nn.LeakyReLU(negative_slope = 0.2))
        layers.append(nn.Dropout(0.3))
        layers.append(nn.Linear(self.hidden3_size, self.hidden2_size))
        layers.append(nn.LeakyReLU(negative_slope = 0.2))
        layers.append(nn.Dropout(0.3))
        layers.append(nn.Linear(self.hidden2_size, self.hidden_size))
        layers.append(nn.LeakyReLU(negative_slope = 0.2))
        layers.append(nn.Dropout(0.3))
        layers.append(nn.Linear(self.hidden_size, 1))
        layers.append(nn.Sigmoid()) #마지막 layer에서는 0 or 1로 판별 -> 0~1의 확률로 표현, Q. 꼭 Sigmoid()를 써야 하는가? -> BCELoss
        self.layersD = nn.Sequential(*layers)
        
    def forward(self, x):
        out = self.layersD(x)
        return out

discriminatorD = DiscriminatorD(image_size, hidden3_size, hidden2_size, hidden_size).to(device)

#optimizer, criterion 설정
optimizerG = optim.Adam(generatorG.parameters(), lr = learning_rate)
optimizerD = optim.Adam(discriminatorD.parameters(), lr = learning_rate)

criterion = nn.BCELoss() #Q.input으로 꼭 sigmoid(), 0~1의 값을 받는가?

# #train
# discriminator_loss = []
# generator_loss = []
# for epoch in range(nb_epochs):
#     #for Batch 학습
#     for idx, [images, labels] in enumerate(train_dataloader):
#         images = images.reshape(batch_size, -1)
#         images = images.to(device)
#         labels = labels.to(device)
#         # print(images.shape) #[128, 1, 28, 28]

#         #labeling
#         real_label = torch.ones(batch_size, 1).to(device)
#         fake_label = torch.zeros(batch_size, 1).to(device)
        
#         #Discriminator 학습 : real image는 1로, fake image는 0으로 판별
#         #input : real image
#         real_image = images
#         real_out = discriminatorD(real_image)
#         real_loss = criterion(real_out, real_label)
        
#         #input : fake image
#         z = torch.randn(batch_size, noise_dim).to(device)
#         fake_image = generatorG(z)
#         fake_out = discriminatorD(fake_image)
#         fake_loss = criterion(fake_out, fake_label)
        
#         lossD = real_loss + fake_loss
        
#         optimizerD.zero_grad()
#         optimizerG.zero_grad()
#         lossD.backward()
#         optimizerD.step()

#         #Generator 학습 : fake image를 1로 판별하게끔
#         #Q. 한 iteration에서, Generator와 Discriminator는 같은 noise로 만들어진 같은 fake image로 train하는 것인가?
#         #Discriminator를 학습할 때 만들었던, 같은 fake image를 써야만 공평한 학습이 이루어진다.
#         # z = torch.randn(batch_size, noise_dim).to(device)
#         fake_image = generatorG(z) #같은 noise를 generator에 input으로 넣으면 계속 같은 fake_image를 만들어 낸다.
#         fake_out = discriminatorD(fake_image)
#         lossG = criterion(fake_out, real_label)
        
#         optimizerG.zero_grad()
#         optimizerD.zero_grad()
#         lossG.backward()
#         optimizerG.step()
        
#     #한 epoch마다
#     discriminator_loss.append(lossD.item())
#     generator_loss.append(lossG.item())
    
#     print('Epoch : {}, Discriminator Loss : {}, Generator Loss : {}'.format(epoch + 1, discriminator_loss[-1], generator_loss[-1]))
#     save_image(fake_image.reshape(batch_size, 1, 28, 28), os.path.join(train_image_dir, 'fake_image_{}.png'.format(epoch + 1)))
    
#모델 파라미터 저장
save_weightG = os.path.join(save_dir, 'generator.pt')
save_weightD = os.path.join(save_dir, 'discriminator.pt')
# torch.save(generatorG.state_dict(), save_weightG)
# torch.save(discriminatorD.state_dict(), save_weightD)

#모델 파라미터 불러오기
generatorG.load_state_dict(torch.load(save_weightG))
discriminatorD.load_state_dict(torch.load(save_weightD))

#Test
with torch.no_grad():
    z = torch.randn(batch_size, noise_dim).to(device)
    #Q. GAN에서의 z는 어떠한 의미를 가질까? Vanilla GAN의 z는 controllable할 수 없는 것 같다. z를 controll할 수 있는 모델은 conditional GAN.
    #같은 z로부터 같은 fake_image가 만들어진다.
    fake_image1 = generatorG(z)
    fake_image1 = fake_image1.reshape(batch_size, 1, 28, 28)
    fake_image2 = generatorG(z)
    fake_image2 = fake_image2.reshape(batch_size, 1, 28, 28)
    cat_image = torch.cat([fake_image1, fake_image2], dim = 3)
    save_image(cat_image, os.path.join(test_image_dir, 'test_image.png'))