import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils

import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torchvision.utils import save_image

import os

#Variational AutoEncoder는 새로운 이미지를 만들 수 있다는 점에서 AutoEncoder와 가장 큰 차이를 보인다. 그렇다면 새로운 이미지는 어떻게 만들어지고, z는 어떠한 의미를 가지는가?
#mode collapse 어떻게 해결?

torch.manual_seed(1234)

device = ('cuda' if torch.cuda.is_available() else 'cpu')

save_image_dir = './VAE_model2/save_image/'
save_model_dir = './VAE_model2/save_model/'

if not os.path.exists(save_image_dir):
    os.makedirs(save_image_dir)
if not os.path.exists(save_model_dir):
    os.makedirs(save_model_dir)
    
#MNIST dataset 가져오기
#Q. Normalize를 안하면 0~1로 값이 나오는 것 같다.
transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize(mean = (0.5, ),
    #                      std = (0.5, ))
])

train_data = dsets.MNIST(root = './MNISTdata', train = True, transform = transform, download = True)
test_data = dsets.MNIST(root = './MNISTdata', train = False, transform = transform, download = True)

#Hyper parameter
batch_size = 256
learning_rate = 0.0001
nb_epochs = 30

#Batch 학습을 위한 Dataloader
train_dataloader = data_utils.DataLoader(dataset = train_data, batch_size = batch_size, shuffle = True, drop_last = True)
test_dataloader = data_utils.DataLoader(dataset = test_data, batch_size = batch_size, shuffle = False, drop_last = True)

# for idx, [images, labels] in enumerate(train_dataloader):
#     print(images) #[0, 1]
#     print(images.shape)
#     break

#Variational AutoEncoder 생성 -> 각 layer는 FCN(MLP)로 이루어져 있다.
#input : [batch_size, 784]
class VAEModel(nn.Module):
    def __init__(self):
        super(VAEModel, self).__init__()
        self.linear1 = nn.Linear(784, 400)
        self.linear2 = nn.Linear(200, 400) #latent size가 model의 성능을 좌지우지 하는 듯, latent size를 2로 했을 때, network는 0, 9, 8만을 학습한다.
        self.linear3 = nn.Linear(400, 784)
        
        self.mean = nn.Linear(400, 200) #Q. 왜 400개의 node에서 2개의 node로 가는가? 2개의 node가 mean을 나타내는 것인가?
        self.logvar = nn.Linear(400, 200)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def encoder(self, x): #input : image, output : mean, variance
        out = self.linear1(x)
        out = self.relu(out)
        mean = self.mean(out)
        logvar = self.logvar(out) #log를 쓰는 이유는 값이 음수가 나올 수 있기 때문에, log 함수의 정의역은 무조건 양수이므로
        return mean, logvar
    
    #Back propagation 시, 왜 reparameterization trick을 써야만 문제가 되지 않는가?
    def reparameterization(self, mean, logvar):
        var = torch.exp(logvar)
        std = torch.sqrt(var)
        eps = torch.randn_like(mean) #mean과 같은 size의 random number들을 tensor에 채워서 반환
        z = mean + eps.mul(std)
        return z
    
    def decoder(self, z):
        out = self.linear2(z)
        out = self.linear3(out)
        recon_x = self.sigmoid(out) #Q. 어떠한 확률?
        return recon_x #0~1
    
    def forward(self, x):
        mean, logvar = self.encoder(x)
        z = self.reparameterization(mean, logvar)
        recon_x = self.decoder(z)
        return recon_x, mean, logvar

model = VAEModel().to(device)

# x = torch.rand(batch_size, 1, 28, 28).to(device)
# print(x)
# print(x.shape)

# x = x.reshape(-1, 784)
# recon_x, mean, logvar = model(x)
# print(recon_x.shape) #[256, 784]
# z = model.reparameterization(mean, logvar)
# print(z.shape) #[256, 2]

# #Optimizer, Criterion 설정
optimizer = optim.Adam(model.parameters(), lr = learning_rate, weight_decay = 1e-5)
#Q. AutoEncoder의 Reconstruction error는 MSE Loss VS VAE의 Reconstruction error는 BCE Loss?
#Loss function에 대해서도 공부해봐야 할듯
#Cross Entropy : x는 관측된 데이터로 고정되어 있고 y를 가장 잘 설명하는 파라미터를 찾고자 할 때,
#VAE의 loss function은 x가 z를 얼마나 잘 inference하는가, z가 x를 얼마나 잘 generate하는가와 관련 -> cross-entropy
#Q. Generative model인데 reconstruction error를 왜 구하지? Train할 때는 같은 image로 reconstruction하는 것이 아닌가?
def criterion(x, recon_x, mean, logvar):
    #reconstruction loss -> GAN과 다르게 
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction = 'sum').to(device) #이미 -부호를 포함하고 있다.
    #F.binary_cross_entropy(input, target, weight = None, size_average = None, reduce = None, reduction = 'mean')
    #input : Tensor of arbitrary shape as probabilities.
    #target : Tensor of the same shape as input with values between 0 and 1.
    #reduction = 'sum' : 모든 loss가 합쳐져서 계산될 것이다. batch_size의 data들 각각의 loss가 더해질 것이다.
    #Q. F.binary_cross_entropy의 reduction = 'mean'의 기본값으로 설정하면, mode collapse의 문제가 발생한다. reduction이 모델에도 많은 영향을 미칠 수 있는가?
    #KL divergence
    KLdiv_loss = 0.5 * torch.sum(mean.pow(2) + logvar.exp() - logvar - 1)
    
    return recon_loss + KLdiv_loss

# recon_loss = F.binary_cross_entropy(x, recon_x) 
# #nn.BCE() : 안에 인수가 없을 때 이용, F.binary_cross_entropy() : 안에 인수가 있을 때 이용
# print(recon_loss) #Tensor
# KLdiv_loss = 0.5 * torch.sum(mean.pow(2) + logvar.exp() - logvar - 1)
# print(KLdiv_loss) #Tensor

# print(len(train_dataloader)) #234 = 전체 data 수 / batch_size -> 한 epoch는 234 iteration을 수행한다.

# #train
# train_loss = []
# for epoch in range(nb_epochs):
#     #배치 학습
#     for idx, [images, labels] in enumerate(train_dataloader):
#         images = images.to(device)
#         images = images.reshape(-1, 784)
#         recon_images, mean, logvar = model(images)
#         loss = criterion(images, recon_images, mean, logvar)
  
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         loss = loss / batch_size
        
#     train_loss.append(loss.item())
#     print('Epoch : {}, Train Loss : {}'.format(epoch + 1, train_loss[-1]))
#     save_image(recon_images.reshape(batch_size, 1, 28, 28), os.path.join(save_image_dir, 'train_image_{}.png'.format(epoch + 1)))
    
#모델 저장
save_weight = os.path.join(save_model_dir, 'save_weights.pt')
# torch.save(model.state_dict(), save_weight)

#모델 불러오기
model.load_state_dict(torch.load(save_weight))

#Test 
#GAN처럼 Generative model이기 때문에, latent space인 z로 새로운 이미지를 생성할 수 있다. 결국 latent space z는 의미있게 되는 것이다.
# with torch.no_grad():
#     for idx, [images, _] in enumerate(test_dataloader):
#         images = images.to(device)
#         images = images.reshape(-1, 784)
#         recon_images, mean, logvar = model(images)
#         # print(images.shape) #[batch_size, 784]
#         # print(recon_images.shape) #[batch_size, 784]
#         images = images.reshape(batch_size, 1, 28, 28)
#         recon_images = recon_images.reshape(batch_size, 1, 28, 28)
#         cat_images = torch.cat([images, recon_images], dim = 3)
        
#         if (idx + 1) % 10 == 0:
#             save_image(cat_images, os.path.join(save_image_dir, 'test_image_{}.png'.format(idx + 1)))
            
#Generative model : latent space z -> new image
with torch.no_grad():
    z = torch.randn(batch_size, 200).to(device)
    recon_x = model.decoder(z)
    save_image(recon_x.reshape(batch_size, 1, 28, 28), os.path.join(save_image_dir, 'z_test_image.png'))