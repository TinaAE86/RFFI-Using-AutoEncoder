import numpy as np
import math
from dataset_gnt.dataset_gnt import *
import pickle as pkl
import os
import sys
from torch.utils.data import Dataset,DataLoader
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from scipy.signal import stft
import matplotlib.pyplot as plt
from scipy.fft import fftshift

def test1_1():
    t_arr = np.linspace(0, 4, 10)
    t_arr1 = np.linspace(0, 4, 10, endpoint=False)
    print(t_arr)
    print(t_arr1)

def test1_2():
    a=np.linspace(0,3,3,endpoint=False)
    b=a**2
    c=np.exp(1j*b) #不能用math.exp
    print(b)
    print(c)

def test1_3():
    a=np.random.uniform(0,1)
    print(a)
    print(a.__class__)

def test1_4():
    a=np.array([1,2,3])
    b=np.square(a)
    print(b)
    print(a*b)

def test1_5():
    a=np.mat(np.array([1j,2+1j,-1j]))
    a_t=np.conjugate(a.transpose())
    sum=a*a_t
    pass

def test1_6():
    a=np.mat([1,2,3])
    a_t=np.transpose(a)
    res=a*a_t
    res1=res.item()
    b=np.array([1])
    res2=b.item()
    pass

def test1_7():
    a=np.random.normal(0,1,3)
    b=1j*np.random.normal(0,1,3)
    c=a+b
    pass

def test1_8():
    root='../dataset'
    file_name='pre_dataset.pkl'
    fr=open(os.path.join(root,file_name),'rb')
    pre_dataset=pkl.load(fr)
    l=pre_dataset.__len__()
    d0=pre_dataset[0]
    fr.close()
    path=sys.path
    pass

def test1_9():
    a=np.array([1+1j,2+1j,3-2j])
    b=a.real
    c=a.imag
    pass

class ComplexDataset(Dataset):
    def __init__(self,data):
        self.data=data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx].real

def test1_10():
    data=np.array([1+1j,2-1j,-1+3j,4-2j])
    # data=[1,2,3,4,5,6]
    cd=ComplexDataset(data)
    cdl=DataLoader(cd,batch_size=2,shuffle=True)
    for batch in cdl:
        pass
    pass

def test1_11():
    s=np.empty((2,5),dtype=float)
    s1=np.random.random(5)
    s2=np.random.random(5)
    s[0]=s1
    s[1]=s2
    t1=torch.tensor(s[0])
    t2=torch.tensor(s[1])
    # t=torch.cat((t1.unsqueeze(0),t2.unsqueeze(0)),dim=0)
    t=torch.stack((t1,t2))
    pass

def test1_12():
    s=[0]
    for _ in range(10):
        s[0]+=1
    pass

def test1_13():
    loss_func=nn.MSELoss()
    t1=torch.randn(3)
    t2=torch.randn(3)
    loss_func_res:torch.Tensor=loss_func(t1,t2)
    loss=loss_func_res.item()
    pass

def test1_14():
    a=np.array([1.0,2,3])
    b=np.inner(a,a)
    pass

def test1_15():
    a=np.array([1,2,3])
    ta=torch.from_numpy(a).unsqueeze(0)
    a+=1
    pass

def test1_16():
    a=np.random.random((4,3))
    ta=torch.from_numpy(a)
    ta0=ta[0]
    ta0_=ta[0].unsqueeze(0)
    a+=1
    pass

def test1_17():
    y=np.array([[1,2,3],[4,5,6]])
    w=np.array([[2],[3]])
    b=np.array([[0.1],[0.2]])
    x=y*w+b
    pass

def test1_18():
    y=np.array([[[1,2,3],
                 [1,2,3]],
                [[1,2,3],
                 [4,5,6]]])
    w=np.array([[[2],
                 [3]],
                [[2],
                 [3]]])
    b=np.array([[[0.1],
                 [0.2]],
                [[0.1],
                 [0.2]]])
    x=y*w+b
    pass

def test1_19():
    x=np.random.random((3,5))
    l=len(x)
    xmin=np.min(x,axis=1).reshape(3,1)
    xmax=np.max(x,axis=1).reshape(3,1)
    y=(x-xmin)/(xmax-xmin)
    pass

def test1_20():
    y=torch.from_numpy(np.array([[[1,2,3],
                 [1,2,3]],
                [[1,2,3],
                 [4,5,6]]]))
    w=torch.from_numpy(np.array([[[2],
                 [3]],
                [[2],
                 [3]]]))
    b=torch.from_numpy(np.array([[[0.1],
                 [0.2]],
                [[0.1],
                 [0.2]]]))
    x=y*w+b
    pass

class HardClamp(nn.Module):
    def __init__(self):
        super(HardClamp,self).__init__()

    def forward(self, x):
        return torch.clamp(x, 0, 1)

def test1_21():
    h=HardClamp()
    a=torch.tensor([-1,-0.5,6,0.3,1,0.4,0])
    out=h(a)
    pass

def test1_22():
    a=np.array([[1,2,3],[2,3,4]])
    b=np.array([1,-1,2])
    c=a*b
    pass

def test1_23():
    a=np.array([[1,2,4],[3,4,9]])
    a_max=np.max(a)
    pass

def test1_24():
    batch_size,channels,N=8,1,64
    a=torch.randn((batch_size,channels,2,N))
    conv1=nn.Conv2d(in_channels=1,out_channels=2,kernel_size=(2,3),padding=(0,1),stride=(1,1))
    a=conv1(a)
    pass

def test1_25():
    a=torch.tensor([1.0,2.0,3.0])
    b=torch.tensor([2.0,3.0,4.0])
    loss_func=nn.MSELoss()
    loss_t:torch.Tensor=loss_func(a,b)
    loss=loss_t.item()
    pass

def test1_26():
    # 生成一个示例信号
    fs=1000  # 采样频率
    t=np.arange(0,2,1/fs)  # 时间向量
    x=np.exp(1j*2*np.pi*50*t)
    x1=x.real

    # 计算STFT
    f,t,Zxx=stft(x,fs=fs,nperseg=256,return_onesided=False)
    f1,t1,Zxx1=stft(x1,fs=fs,nperseg=256,return_onesided=False)

    #移位
    Zxx=fftshift(Zxx,axes=0)
    f=fftshift(f)

    # 绘制STFT结果
    plt.pcolormesh(t,f,np.abs(Zxx),shading='gouraud')
    plt.colorbar()
    plt.title('STFT Magnitude')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()

def test1_27():
    # 生成示例信号
    fs=1000  # 采样频率
    t=np.arange(0,2,1/fs)  # 时间向量
    x=np.sin(2*np.pi*50*t)+np.exp(1j*2*np.pi*120*t)

    # 计算STFT
    f,t,Zxx=stft(x,fs=fs,nperseg=256,return_onesided=False)

    # 使用fftshift将频率轴和Zxx进行移位
    Zxx_shifted=fftshift(Zxx,axes=0)
    f_shifted=fftshift(f)

    # 绘制STFT结果（双边谱）
    plt.pcolormesh(t,f_shifted,np.abs(Zxx_shifted),shading='gouraud')
    plt.colorbar()
    plt.title('STFT Magnitude (Two-sided Spectrum)')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()

if __name__=='__main__':
    test1_26()