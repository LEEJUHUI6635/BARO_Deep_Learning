import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils

import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torchvision.utils import save_image

import os

#Variational AutoEncoder -> Generative Model : GAN의 z 특성 + AutoEncoder의 reconstruction 특성
#새로운 이미지는 어떻게 만들어지고, z는 어떠한 의미를 가지는가?
#latent size = 2로 설정하였을 때, mode collapse의 문제가 발생하였다. 이는 latent space가 매우 작기 때문,
torch.manual_seed(1234)

device = ('cuda' if torch.cuda.is_available() else 'cpu')
save_model_dir = './VAE_model/save_model/'
save_train_dir = './VAE_model/save_train_image/'
save_test_dir = './VAE_model/save_test_image/'

if not os.path.exists(save_model_dir):
    os.makedirs(save_model_dir)
if not os.path.exists(save_train_dir):
    os.makedirs(save_train_dir)
if not os.path.exists(save_test_dir):
    os.makedirs(save_test_dir)
    
#MNIST dataset 가져오기
#Normalize를 하지 않으면, 0~1의 값으로 나온다. 
#Q. Normalize는 dataset을 어떻게 변화시키는가?
transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize(mean = 0.5,
    #                      std = 0.5)
])
train_data = dsets.MNIST(root = './MNISTdata', train = True, transform = transform, download = True)
test_data = dsets.MNIST(root = './MNISTdata', train = False, transform = transform, download = True)

#Hyper parameter
batch_size = 128
learning_rate = 0.0001
input_size = 784
hidden_size = 400
latent_size = 200 #Q. latent size가 train 결과에 어떠한 영향을 미치는가?
#latent space의 node가 적다는 것은 그만큼 학습하는 feature의 개수가 적은 것이기 때문에, 성능에 큰 영향을 줄 수 있는 것인가?
nb_epochs = 30

#Batch 학습을 위한 Dataloader
train_dataloader = data_utils.DataLoader(dataset = train_data, batch_size = batch_size, shuffle = True, drop_last = True)
test_dataloader = data_utils.DataLoader(dataset = test_data, batch_size = 1, shuffle = False, drop_last = False)

#Model 생성 
#VAE Model -> 각 layer는 FCN(MLP)로 이루어져 있다.
class VAEModel(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size):
        super(VAEModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size #latent size가 model의 성능을 좌지우지 하는 듯, latent size를 2로 했을 때, network는 1, 0, 9, 8만을 학습한다.
        #Encoder
        self.linearE1 = nn.Linear(self.input_size, self.hidden_size)
        self.linearE2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.mean = nn.Linear(self.hidden_size, self.latent_size)
        self.logvar = nn.Linear(self.hidden_size, self.latent_size) #logvar : log 함수의 정의역 자체가 양수이기 때문, 
        #Decoder
        self.linearD1 = nn.Linear(self.latent_size, self.hidden_size)
        self.linearD2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.linearD3 = nn.Linear(self.hidden_size, self.input_size)
        
        self.sigmoid = nn.Sigmoid()
        self.leakyrelu = nn.LeakyReLU(0.2)
        
    def encoder(self, x):
        out = self.leakyrelu(self.linearE1(x))
        out = self.leakyrelu(self.linearE2(out))
        mean = self.mean(out)
        logvar = self.logvar(out)
        return mean, logvar
    
    #Q. Backprop 시, 왜 reparameterization trick을 써야만 문제가 되지 않는가?
    def reparameterization(self, mean, logvar):
        var = torch.exp(logvar)
        std = torch.sqrt(var)
        eps = torch.randn_like(std) #torch.randn_like() : std와 같은 size의 random number를 tensor에 채워서 반환한다.
        z = mean + eps.mul(std)
        return z        
    
    def decoder(self, z):
        out = self.leakyrelu(self.linearD1(z))
        out = self.leakyrelu(self.linearD2(out)) #leakyrelu -> Generation의 task이기 때문
        out = self.linearD3(out)           
        recon_x = self.sigmoid(out) #마지막 activation function이 sigmoid가 되어야 하는 이유는 처음 dataset을 받아올 때 0~1로 mapping되어 input으로 들어오기 때문
        return recon_x #Q. 어떠한 확률 값인가?
    
    def forward(self, x):
        mean, logvar = self.encoder(x)
        z = self.reparameterization(mean, logvar)
        recon_x = self.decoder(z)
        return recon_x, mean, logvar
    
model = VAEModel(input_size, hidden_size, latent_size).to(device)

#Optimizer, Criterion 설정
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

#Q. AutoEncoder의 Reconstruction error는 MSE Loss VS VAE의 Reconstruction error는 BCE Loss?
#Loss function에 대해서도 공부해 봐야 할 듯
#Cross Entropy : x는 관측된 데이터로 고정되어 있고, y를 가장 잘 설명하는 파라미터를 찾고자 할 때,
#VAE의 loss function은 x가 z를 얼마나 잘 inference하는가, z가 x를 얼마나 잘 generate하는가와 관련 -> cross-entropy
#Q. Generative Model인데 reconstruction error를 왜 구하지? Train할 때에는 같은 image로 reconstruction하는 것이 아닌가?
#VAE의 loss
def criterion(x, recon_x, mean, logvar):
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction = 'sum') 
    #nn.BCE() : 안에 인수가 없을 때 이용, F.binary_cross_entropy() : 안에 인수가 있을 때 이용
    #F.binary_cross_entropy(input, target, weight = None, size_average = None, reduce = None, reduction = 'mean')
    #input : Tensor of arbitrary shape as probabilities.
    #target : Tensor of the same shape as input with values between 0 and 1.
    #reduction = 'sum' : 모든 loss가 합쳐져서 계산될 것이다. batch_size의 data들 각각의 loss가 더해질 것이다.
    #Q. F.binary_cross_entropy의 reduction = 'mean'의 기본값으로 설정하면, 모델에도 많은 영향을 미치는가?
    #A. 학습이 제대로 되지 않는 것을 확인할 수 있다.
    KL_loss = 0.5 * torch.sum(mean.pow(2) + logvar.exp() - logvar - 1)
    return recon_loss + KL_loss

#Train
train_loss = []
for epoch in range(nb_epochs):
    #Batch 학습
    for idx, [images, _] in enumerate(train_dataloader):
        images = images.reshape(-1, 784)
        images = images.to(device)

        recon_images, mean, logvar = model(images)
        
        loss = criterion(images, recon_images, mean, logvar)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss = loss / batch_size
        
    #한 epoch 마다,
    train_loss.append(loss.item())
    print('Epoch : {}, Train Loss : {}'.format(epoch + 1, train_loss[-1]))
    save_image(recon_images.reshape(batch_size, 1, 28, 28), os.path.join(save_train_dir, 'train_image_{}.png'.format(epoch + 1)))
    
#모델 parameter 저장
save_weight = os.path.join(save_model_dir, 'save_weights.pt')
torch.save(model.state_dict(), save_weight)

#모델 parameter 불러오기
model.load_state_dict(torch.load(save_weight))

# print(len(test_dataloader)) #10000

# #test -> reconstruction image
# with torch.no_grad():
#     for idx, [images, _] in enumerate(test_dataloader):
#         images = images.reshape(-1, 784)
#         images = images.to(device)
        
#         recon_images, mean, logvar = model(images)
#         images = images.reshape(1, 1, 28, 28)
#         recon_images = recon_images.reshape(1, 1, 28, 28)
#         cat_images = torch.cat([images, recon_images], dim = 3)
#         if (idx + 1) % 1000 == 0:
#             save_image(cat_images, os.path.join(save_test_dir, 'test_image_{}.png'.format(idx + 1)))

# #test -> new image : latent space에서 새로운 image 생성
# #GAN처럼 Generative Model이기 때문에, latent space인 z로부터 새로운 이미지를 생성할 수 있다. 결국 latent space z는 의미있게 되는 것이다.
# with torch.no_grad():
#     z = torch.randn(batch_size, latent_size).to(device)
#     recon_x = model.decoder(z)
#     recon_x = recon_x.reshape(batch_size, 1, 28, 28)
#     save_image(recon_x, os.path.join(save_test_dir, 'z_test_image.png'))