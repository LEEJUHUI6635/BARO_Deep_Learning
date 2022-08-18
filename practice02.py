import torch
import pandas as pd
import matplotlib
import numpy as np
import sklearn

print('Hello baro')

num = 3
print(num, type(num)) #int

real_value = 1.0
print(real_value, type(real_value)) #float

string = 'data'
print(string, type(string)) #str

#배열 -> 리스트
baro_list = [1, 2, 3, 4, 5]
print(type(baro_list)) #list
print(baro_list)

add_value = 6
baro_list.append(add_value)
print(baro_list)

#리스트의 크기 -> len
print(len(baro_list)) #6

baro_list = ['하나', '둘', '셋', '넷', '다섯']
for index, value in enumerate(baro_list):
    print(index)
    print(value)
    break

num = np.array(3)
print(num)
print(type(num)) #array

string = np.array('data')
print(string)
print(type(string)) #array

arr_onerank = np.array([1, 2, 3])
print(arr_onerank)
print(type(arr_onerank)) #array

arr_matrix = np.array([[1, 2, 3], [4, 5, 6]])
print(arr_matrix)
print(type(arr_matrix)) #array

#배열의 크기 -> shape
print(num.shape) #() -> scalar
print(arr_onerank.shape) #(3, )
print(arr_matrix.shape) #(2, 3)
print(arr_matrix)

arr_reshape = arr_matrix.reshape(-1) #"-1"은 나머지 성분을 차례로 배열
print(arr_reshape.shape) #(6, )
print(arr_reshape)

np.random.seed(777)
linspace_arr = np.linspace(0, 10, 5) #0~10까지 5개의 숫자를 만들어 낸다. 간격 자동 설정
print(linspace_arr)

rand_arr = np.random.rand(10) #0~1까지 10개의 난수 생성
print(rand_arr)

normal_arr = np.random.randn(1, 2) #[1, 2] size의 평균 0, 표준 정규 분포를 따르는 난수 생성
print(normal_arr)

arange_arr = np.arange(1, 10, 2) #start, stop, step
print(arange_arr)

arange_arr2 = np.arange(0, 5, 1) #항상 끝 값은 포함하지 않는다.
print(arange_arr2)

arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
arr3 = np.concatenate([arr1, arr2])
print(arr3)

arr1 = np.arange(3).reshape(1, 3) #[0, 1, 2] -> [[0 1 2]]
arr2 = np.arange(6).reshape(2, 3) #[0, 1, 2, 3, 4, 5] -> [[0 1 2] [3 4 5]]
arr_concat = np.concatenate([arr1, arr2], axis = 0)
print(arr_concat.shape)
print(arr_concat)

#Pandas
df_raw = pd.read_csv('./HousingData.csv')
print(df_raw.shape)
print(df_raw)

#Column 삭제하기
df_raw_new = df_raw.drop("CRIM", axis = 1)
print(df_raw_new.head()) #상위 5개까지만

#Tensor
t = torch.FloatTensor([0., 1., 2., 3., 4., 5., 6.])
print(t)
print(t.dim()) #dim = 1
print(t.shape) #[7]
print(t.size()) #[7]

torch.manual_seed(777)

#가우시안 분포
x = torch.rand(2, 3) #0과 1 사이의 숫자를 랜덤으로 반환
print(x)

x = torch.randint(10, size = (2, 3)) #지정 범위 내 숫자를 랜덤으로 반환, 0 ~ 10
print(x)

x = torch.randn(2, 3) #가우시안 정규 분포를 따르는 수를 랜덤 변환
print(x)

x = torch.zeros(2, 3)
print(x)

x = torch.ones(2, 3)
print(x)

x = torch.arange(0, 3, 0.5) #start, stop, step
print(x)

x = torch.FloatTensor(2, 3)
print(x)

#list로부터 2x3 텐서 생성
x_list = [[1, 2, 3], [4, 5, 6]]
x = torch.Tensor(x_list)
print(x)

#numpy array로부터 2x3 텐서 생성
x_numpy = np.array([[1, 2, 3], [4, 5, 6]])
x = torch.Tensor(x_numpy)
print(x)

#Tensor 크기 변경하기
x = np.array([[[0, 1, 2],
               [3, 4, 5]],
              [[6, 7, 8],
               [9, 10, 11]]])
x = torch.FloatTensor(x)
print(x.shape) #[2, 2, 3] -> 3개의 숫자가 2개가 있고, 또 그것이 2개가 있다. 뒤에서부터 읽으면 편함.

tensor_x = x.view(-1, 3)
print(tensor_x.shape) #[4, 3]

tensor_x = x.reshape(-1, 2)
print(tensor_x.shape) #[6, 2]

#Tensor의 차원 추가 및 삭제
x1 = torch.FloatTensor(10, 1, 3, 1, 4)

x2 = x1.squeeze()

print(x1.size()) #[10, 1, 3, 1, 4] 
print(x2.size()) #[10, 3, 4]

#Unsqueeze
x1 = torch.FloatTensor(10, 3, 4)
x2 = x1.unsqueeze(dim = 1)
print(x1.size())
print(x2.size())

#Tensor 합치기
torch.manual_seed(777)
x = torch.randint(10, size = (2, 3))
y = torch.randint(10, size = (2, 3))

z = torch.cat([x, y], dim = 0)

print(x)
print(y)
print(z.shape) #[4, 3]
print(z)

#평균, 분산 구하기
x = torch.FloatTensor([1, 100])

print(x)
print(x.mean())
print(x.var())

#Tensor 인덱싱
print(z)
print(z[3])
print(z[0, 1])
print(z[:, 1])
print(z[-1]) #맨 마지막