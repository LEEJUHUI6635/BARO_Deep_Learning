import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils

import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torchvision.utils import save_image

import os

torch.manual_seed(1234)

device = ('cuda' if torch.cuda.is_available() else 'cpu')

save_model_dir = './DCGAN_model/save_weight'
save_train_dir = './DCGAN_model/train_image'
save_test_dir = './DCGAN_model/test_image'

if not os.path.exists(save_model_dir):
    os.makedirs(save_model_dir)
if not os.path.exists(save_train_dir):
    os.makedirs(save_train_dir)
if not os.path.exists(save_test_dir):
    os.makedirs(save_test_dir)
    
#MNIST dataset에 대한 DCGAN -> CNN(3차원)
#Dataset 불러오기
transform = transforms.Compose([
    transforms.Resize(size = (64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean = (0.5, ), std = (0.5, )),
])

train_data = dsets.MNIST(root = './GAN_MNISTdata', train = True, transform = transform, download = True)

#Hyper parameter
batch_size = 128
learning_rate = 0.0002
noise_dim = 100
hidden_size = 1024 #channel
hidden2_size = 512
hidden3_size = 256
hidden4_size = 128
nb_epochs = 50

#Batch 학습을 위한 Dataloader
train_dataloader = data_utils.DataLoader(dataset = train_data, batch_size = batch_size, shuffle = True, drop_last = True)

#generator 생성
#input : noise z, output : image -> CNN의 upsampling을 이용한다. -> deconvolution 연산 사용
class GeneratorG(nn.Module):
    def __init__(self, noise_dim, hidden_size, hidden2_size, hidden3_size, hidden4_size):
        super(GeneratorG, self).__init__()
        self.noise_dim = noise_dim
        self.hidden_size = hidden_size
        self.hidden2_size = hidden2_size
        self.hidden3_size = hidden3_size
        self.hidden4_size = hidden4_size
        layers = []
        #ConvTranspose2d : stride = 2 -> image size 2배 확대, 공식에 대해 조금 더 찾아봐야 할 듯.
        #ConvTranspose2d : out = stride * (in - 1) + kernel_size - 2 * padding
        layers.append(nn.ConvTranspose2d(in_channels = self.noise_dim, out_channels = self.hidden_size, kernel_size = 4, stride = 1, padding = 0, bias = False)) #Q.bias = False? -> 단순히 연산량을 줄이기 위해?
        #out = 1*(1-1) + 4 - 2*0 = 4
        layers.append(nn.BatchNorm2d(num_features = self.hidden_size))
        layers.append(nn.ReLU()) #Q.image generation(reconstruction)인데, 왜 LeakyReLU가 아닌 ReLU?
        layers.append(nn.ConvTranspose2d(in_channels = self.hidden_size, out_channels = self.hidden2_size, kernel_size = 4, stride = 2, padding = 1, bias = False))
        #out = 2*(4-1) + 4 - 2*1 = 8 
        layers.append(nn.BatchNorm2d(num_features = self.hidden2_size))
        layers.append(nn.ReLU())
        layers.append(nn.ConvTranspose2d(in_channels = self.hidden2_size, out_channels = self.hidden3_size, kernel_size = 4, stride = 2, padding = 1, bias = False))
        #out = 2*(8-1) + 4 - 2*1 = 16
        layers.append(nn.BatchNorm2d(num_features = self.hidden3_size))
        layers.append(nn.ReLU())
        layers.append(nn.ConvTranspose2d(in_channels = self.hidden3_size, out_channels = self.hidden4_size, kernel_size = 4, stride = 2, padding = 1, bias = False))
        #out = 2*(16-1) + 4 - 2*1 = 32
        layers.append(nn.BatchNorm2d(num_features = self.hidden4_size))
        layers.append(nn.ReLU())
        layers.append(nn.ConvTranspose2d(in_channels = self.hidden4_size, out_channels = 1, kernel_size = 4, stride = 2, padding = 1, bias = False))
        #out = 2*(32-1) + 4 - 2*1 = 64
        layers.append(nn.Tanh()) #Q.마지막 layer에 대해서 Batch Normalization을 하지 않는 이유?
        #Tanh() : Dataloader로 image를 받아올 때, -1~1로 mapping되어 나오기 때문
        self.layersG = nn.Sequential(*layers)
        
    def forward(self, x):
        out = self.layersG(x)
        return out
        
generatorG = GeneratorG(noise_dim, hidden_size, hidden2_size, hidden3_size, hidden4_size).to(device)

#noise 생성 -> 3차원인 CNN에서 generation을 수행해야하기 때문에, input은 4차원 Tensor여야 한다. image size(image width, image height) 부분이 추가되어야 한다.
# z = torch.randn(batch_size, noise_dim, 1, 1).to(device)
# fake_image = generatorG(z)
# print(fake_image.shape) #[128, 1, 64, 64]

#discriminator 생성
#input : image, output : classification -> CNN의 downsampling을 이용한다.
class DiscriminatorD(nn.Module):
    def __init__(self, hidden4_size, hidden3_size, hidden2_size, hidden_size, batch_size):
        super(DiscriminatorD, self).__init__()
        self.hidden4_size = hidden4_size
        self.hidden3_size = hidden3_size
        self.hidden2_size = hidden2_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        layers = [] 
        #Convolution 연산 공식 : out = (in + 2 * padding - kernel_size) / stride + 1
        layers.append(nn.Conv2d(in_channels = 1, out_channels = self.hidden4_size, kernel_size = 4, stride = 2, padding = 1, bias = False)) #bias = False -> bias는 학습하지 않겠다.
        #out = (64+2*1-4)/2 + 1 = 32
        layers.append(nn.BatchNorm2d(num_features = self.hidden4_size)) #pre-activaton -> Batch Normalization
        layers.append(nn.LeakyReLU(negative_slope = 0.2))
        layers.append(nn.Conv2d(in_channels = self.hidden4_size, out_channels = self.hidden3_size, kernel_size = 4, stride = 2, padding = 1, bias = False))
        #out = (32+2*1-4)/2 + 1 = 16
        layers.append(nn.BatchNorm2d(num_features = self.hidden3_size))
        layers.append(nn.LeakyReLU(negative_slope = 0.2), inplace = True) #inplace = True -> memory save, Q. 왜 다른 layer에서는 inplace를 하지 않는가?
        layers.append(nn.Conv2d(in_channels = self.hidden3_size, out_channels = self.hidden2_size, kernel_size = 4, stride = 2, padding = 1, bias = False))
        #out = (16+2*1-4)/2 + 1 = 8
        layers.append(nn.BatchNorm2d(num_features = self.hidden2_size))
        layers.append(nn.LeakyReLU(negative_slope = 0.2), inplace = True)
        layers.append(nn.Conv2d(in_channels = self.hidden2_size, out_channels = self.hidden_size, kernel_size = 4, stride = 2, padding = 1, bias = False))
        #out = (8+2*1-4)/2 + 1 = 4
        layers.append(nn.BatchNorm2d(num_features = self.hidden_size))
        layers.append(nn.LeakyReLU(negative_slope = 0.2), inplace = True)
        layers.append(nn.Conv2d(in_channels = self.hidden_size, out_channels = 1, kernel_size = 4, stride = 1, padding = 0, bias = False))
        #out = (4+2*0-4)/1 + 1 = 1
        layers.append(nn.Sigmoid())       
        
        self.layersD = nn.Sequential(*layers)
        
    def forward(self, x):
        out = self.layersD(x)
        return out
    
discriminatorD = DiscriminatorD(hidden4_size, hidden3_size, hidden2_size, hidden_size, batch_size).to(device)

#Weight 초기화 -> Convolution, TransConvolution에 대한 weight, Batch Normalization에 대한 weight
def weight_initialization(model):
    func_name = model.__class__.__name__
    if func_name.find('Conv') != -1: #Convolution, TransConvolution은 bias를 학습하지 않는다.
        nn.init.normal_(model.weight.data, mean = 0.0, std = 0.02)
    elif func_name.find('BatchNorm') != -1:
        nn.init.normal_(model.weight.data, mean = 1.0, std = 0.02)
        nn.init.constant_(model.bias.data, val = 0.0)
        
generatorG.apply(weight_initialization)
discriminatorD.apply(weight_initialization)

#optimizer, criterion 설정
optimizerG = optim.Adam(generatorG.parameters(), lr = learning_rate)
optimizerD = optim.Adam(discriminatorD.parameters(), lr = learning_rate)

criterion = nn.BCELoss()

#Train
discriminator_loss = []
generator_loss = []

for epoch in range(nb_epochs):
    #배치 학습
    for idx, [images, labels] in enumerate(train_dataloader):
        #real label과 fake label 생성
        real_label = torch.ones(batch_size, 1).to(device)
        fake_label = torch.zeros(batch_size, 1).to(device)
        
        #Discriminator 학습
        #real image
        real_image = images.to(device)
        real_predict = discriminatorD(real_image)
        # print(real_predict.shape) #[128, 1, 1, 1]
        real_predict = real_predict.reshape(batch_size, -1)
        real_loss = criterion(real_predict, real_label)
    
        #fake image
        z = torch.randn(batch_size, noise_dim, 1, 1).to(device)
        fake_image = generatorG(z)
        fake_predict = discriminatorD(fake_image)
        # print(fake_predict.shape)
        fake_predict = fake_predict.reshape(batch_size, -1)
        # print(fake_predict.shape)
        fake_loss = criterion(fake_predict, fake_label)
        
        lossD = (real_loss + fake_loss) / 2
        
        # print(lossD)
        
        optimizerD.zero_grad()
        optimizerG.zero_grad()
        lossD.backward()
        optimizerD.step()
        
        #Generator 학습
        #Discriminator의 학습에 이용한 같은 fake image로 학습해야 공평한 학습이 이루어진다.
        fake_image = generatorG(z)
        fake_predict = discriminatorD(fake_image)
        # print(fake_predict.shape)
        fake_predict = fake_predict.reshape(batch_size, -1)

        lossG = criterion(fake_predict, real_label)
        
        # print(lossG)
    
        optimizerG.zero_grad()
        optimizerD.zero_grad()
        lossG.backward()
        optimizerG.step()

    #한 epoch마다,
    discriminator_loss.append(lossD.item())
    generator_loss.append(lossG.item())
    print('Epoch : {}, Discriminator Loss : {}, Generator Loss : {}'.format(epoch + 1, discriminator_loss[-1], generator_loss[-1]))
    save_image(fake_image, os.path.join(save_train_dir, 'train_image_{}.png'.format(epoch + 1)))

#모델 파라미터 저장
save_weightG = os.path.join(save_model_dir, 'generator.pt')
save_weightD = os.path.join(save_model_dir, 'discriminator.pt')
# torch.save(generatorG.state_dict(), save_weightG)
# torch.save(discriminatorD.state_dict(), save_weightD)

#모델 파라미터 불러오기
generatorG.load_state_dict(torch.load(save_weightG))
discriminatorD.load_state_dict(torch.load(save_weightD))

#Test
with torch.no_grad():
    z = torch.randn(batch_size, noise_dim, 1, 1).to(device)
    fake_image = generatorG(z)
    print(fake_image.shape) #[128, 1, 64, 64]
    save_image(fake_image, os.path.join(save_test_dir, 'test_image.png'))
    