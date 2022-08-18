import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(1234)

device = ('cuda' if torch.cuda.is_available() else 'cpu')

#Vanilla RNN -> 다음 글자를 예측한다.

#Data Preprocessing
string = 'tomato'
# print(string)
string = list(string)
# print(string) #['t', 'o', 'm', 'a', 't', 'o']
# print(string[0])

set_string = set(string) #set : 중복을 허용하지 않는 data type
# print(set_string) #{'m', 'o', 't', 'a'}
# print(type(set_string)) #class 'set'

set_string = ['t', 'o', 'm', 'a']
# print(set_string)
# print(type(set_string)) #list

#Data -> One-hot Encoding
#Data x는 마지막 글자를 뺀다.
#괄호를 하나 더 만들어 준 이유는 list를 Tensor로 변환했을 때, [seq, batch, feature]의 형태로 만들기 위해서,
x_train = [[[1, 0, 0, 0], #t
           [0, 1, 0, 0], #o
           [0, 1, 1, 0], #m
           [0, 0, 0, 1], #a
           [1, 0, 0, 0]]] #t

# print(x_train)
# print(type(x_train)) #list 

#Data y는 첫 번째 글자를 뺀다. 
#Data y는 index로 구성된다. -> classification의 문제 -> Cross-Entropy Loss(Softmax + argmax를 포함)
y_train = [[1, 2, 3, 0, 1]]

# print(y_train)
# print(type(y_train)) #list

x_train_tensor = torch.FloatTensor(x_train)
y_train_tensor = torch.LongTensor(y_train)

# print(x_train_tensor)
# print(x_train_tensor.shape) #[1, 5, 4] -> [batch, seq, feature]
# print(y_train_tensor)
# print(y_train_tensor.shape) #[1, 5]

#Hyper parameter
#[seq, batch, feature]
input_size = 4 #variable length, feature의 개수
hidden_size = 4 #Q. hidden_size는 input_size와 같아야 하지 않나? 혹은 hidden_layer의 개수인가?
learning_rate = 0.1
nb_epochs = 20

#Model 설정
#nn.RNN(input_size, hidden_size, num_layers, nonlinearity, bias, batch_first, dropout, bidirectional)
#input_size : The number of expected features in the input x.
#hidden_size : The number of features in the hidden state h.
#num_layers : Number of recurrent layers. Default: 1
#nonlinearity : The non-linearity to use. Can be either 'tanh' or 'relu'. Default: 'tanh'
#bias : If False, the the layer does not use bias weights b_ih and b_hh. Default: True
#batch_first : If True, the the input and output tensors are provided as (batch, seq, feature) instead of (seq, batch, feature).
#Note that this does not apply to hidden or cell states. Default: False
#bidirectional : If True, becomes a bidirectional RNN. Default: True
model = nn.RNN(input_size, hidden_size, batch_first = True).to(device)
#batch_first = True -> [batch, seq, feature]
#Model의 output
#output : tensor of shape (L, D*Hout) for unbatched input, (L, N, D*Hout) for batched input (batch_first = False)
#h_n : tensor of shape (D*num_layers, Hout) for unbatched input or (D*num_layers, N, Hout) for batched input
#Hout = hidden_size
#D = 2 if bidirectional else 1
x_train_tensor = x_train_tensor.to(device)
output, h_n = model(x_train_tensor)
# print(output)
# print(type(output)) #Tensor
# print(output.shape) #[1, 5, 4] -> [batch_size, seq, feature]

# print(h_n)
# print(type(h_n)) #Tensor
# print(h_n.shape) #[1, 1, 4]

#Optimizer, Criterion 설정
optimizer = optim.Adam(model.parameters(), lr = learning_rate)
criterion = nn.CrossEntropyLoss() #결국, classification의 문제이다.
#output = loss(input, target)

#Train -> 배치 학습 x
for epoch in range(nb_epochs):
    output, h_n = model(x_train_tensor)
    # output = output.to(device)
    y_train_tensor = y_train_tensor.to(device)
    # print(output)
    # print(output.shape) #[1, 5, 4]
    # print(y_train_tensor.shape) #[1, 5]
    # print(y_train_tensor.reshape(-1)) #[1, 2, 3, 0, 1]
    # print((y_train_tensor.reshape(-1)).shape)
    output = output.reshape(-1, input_size)
    y_train_tensor = y_train_tensor.reshape(-1)
    loss = criterion(output, y_train_tensor)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    #softmax + argmax -> 가장 큰 확률의 index 추출
    output_softmax = torch.softmax(output, dim = 1)
    # print(output_softmax)
    output_argmax = torch.argmax(output, dim = 1)
    output_argmax = output_argmax.cpu().detach().numpy() #cpu -> numpy 형태
    # print(output_argmax) #[2, 3, 2, 2, 2]
    result = ''.join([set_string[s] for s in output_argmax])
    print('Epoch : {}, Train Loss : {}, result_argmax : {}, result : {}'.format(epoch + 1, loss.item(), output_argmax, result))
    
#Test
x_test = torch.FloatTensor([[1, 0, 0, 0],
                            [0, 1, 0, 0]])
print(x_test)
print(x_test.shape) #[2, 4]
x_test = x_test.reshape(1, 2, 4) #batch_size, length, feature

with torch.no_grad():
    output = output.to(device)
    x_test = x_test.to(device)
    output, h_n = model(x_test)
    output = output.reshape(-1, input_size) #input_size = feature
    output_softmax = torch.softmax(output, dim = 1)
    output_argmax = torch.argmax(output, dim = 1)
    output_argmax = output_argmax.cpu().detach().numpy()
    result = ''.join([set_string[s] for s in output_argmax])
    print('result_argmax : {}, result : {}'.format(output_argmax, result)) #om