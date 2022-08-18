import torch 
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils

import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torchvision.utils import save_image

import os

#AutoEncoder -> Unsupervised Learning, Vanilla AutoEncoder는 reconstruction만을 수행한다.
#loss function -> reconstruction error
#AutoEncoder와 VAE와의 큰 차이점 : AutoEncoder는 'prior에 대한 조건(Condition)'이 없기 때문에 의미있는 z vector의 space가 계속해서 바뀌게 된다. 

torch.manual_seed(1234)

device = ('cuda' if torch.cuda.is_available() else 'cpu')

save_dir = './FashionMNIST/'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    
#Fashion MNIST dataset 불러오기
#Q. Normalize는 왜 안하는가? model의 마지막 layer에 sigmoid 함수를 사용하는 것과 관련이 있는가?

transform = transforms.Compose([
    transforms.ToTensor(), #[0, 1]로 mapping한다.
    # transforms.Normalize(mean = (0.5, ),
    #                      std = (0.5, ))
])

train_data = dsets.FashionMNIST(root = './FashionMNISTdata', train = True, transform = transform, download = True)
test_data = dsets.FashionMNIST(root = './FashionMNISTdata', train = False, transform = transform, download = True)

#Hyper parameter
batch_size = 256
learning_rate = 0.0001
image_size = 784
latent_size = 16 #latent size는 dataset이 가진 feature의 개수인가?
nb_epochs = 100

#Batch 학습을 위한 Dataloader
train_dataloader = data_utils.DataLoader(dataset = train_data, batch_size = batch_size, shuffle = True, drop_last = True)
test_dataloader = data_utils.DataLoader(dataset = test_data, batch_size = 1, shuffle = False, drop_last = False)

# for idx, [images, _] in enumerate(train_dataloader):
#     print(images)
#     save_image(images, os.path.join(save_dir, 'batch_image.png'))
#     break
    # print(images.shape) #[batch_size, 1, 28, 28]

#AutoEncoder model 생성 -> Encoder와 Decoder 모두 perceptron 이용, image [batch_size, 1, 28, 28] -> [batch_size, 784]
class AutoEncoderModel(nn.Module):
    def __init__(self, image_size, latent_size):
        super(AutoEncoderModel, self).__init__()
        self.image_size = image_size
        self.latent_size = latent_size
        self.encoder = nn.Linear(self.image_size, self.latent_size) #Encoder를 통해 latent space z 생성 -> 입력의 특징을 16으로 압축한다. Q. 숫자의 의미가 따로 있는가? 어떻게 수치를 정해야 하는가?
        self.decoder = nn.Linear(self.latent_size, self.image_size) #Decoder를 통해 latent space를 가지고 reconstruction image 생성
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        out = self.sigmoid(out) #[0, 1] -> input image와의 pixel 범위 맞춰준다.
        return out
    
model = AutoEncoderModel(image_size, latent_size).to(device)

#optimizer, criterion 설정
optimizer = optim.Adam(model.parameters(), lr = learning_rate)
criterion = nn.MSELoss() #Q. Reconstruction error를 구해야 하기 때문에, input image pixel 값 - recon image pixel 값?

# #Train
# train_loss = [] 
# for epoch in range(nb_epochs):
#     #Batch 학습
#     for idx, [images, _] in enumerate(train_dataloader):
#         images = images.reshape(batch_size, -1)
#         images = images.to(device)
        
#         recon_images = model(images)
        
#         loss = criterion(recon_images, images)
        
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
        
#     #한 epoch 마다,
#     train_loss.append(loss.item())
    
#     print('Epoch : {}, Train Loss : {}'.format(epoch + 1, train_loss[-1]))
#     save_image(recon_images.reshape(batch_size, 1, 28, 28), os.path.join(save_dir, 'train_image_{}.png'.format(epoch + 1)))

#Model parameter 저장
save_weight = os.path.join(save_dir, 'save_weights.pt')
# torch.save(model.state_dict(), save_weight)

#Model 불러오기
model.load_state_dict(torch.load(save_weight))

# print(len(test_data)) #10000

#Test
with torch.no_grad():
    for idx, [images, _] in enumerate(test_dataloader): #batch_size = 10000
        images = images.reshape(1, -1)
        # print(images.shape) #[10000, 784]
        images = images.to(device)
        recon_images = model(images)
        # print(recon_images.shape) #[10000, 784]
        images = images.reshape(1, 1, 28, 28)
        recon_images = recon_images.reshape(1, 1, 28, 28)
        cat_images = torch.cat([images, recon_images], dim = 3)
        # print(cat_images.shape) #[1, 1, 28, 56]
        if (idx + 1) % 100 == 0:
            save_image(cat_images, os.path.join(save_dir, 'test_image_{}.png'.format(idx + 1)))
            #test image와 latent space z로부터 만들어낸 reconstruction image는 일치한다. 결국, AutoEncoder는 generative model이 아닌 reconstruction의 task이다.
