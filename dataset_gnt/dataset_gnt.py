import numpy as np
import torch
from torch.utils.data import Dataset
from . import signal_controller_base
from .lora_signal_controller import LoraSigCtrl
from .device_controller import *
from typing import List,Optional
import pickle as pkl
import os
import math
from tqdm import tqdm,trange
from model.ae import *

pi=math.pi

class PreDataset(Dataset):
    '''
    归一化操作放在网络中，使用Sigmoid函数归一化
    '''
    def __init__(self,t_arr:np.ndarray,sb:np.ndarray,rb:np.ndarray,tensor:bool=False):
        self.t_arr=t_arr
        self.sb=sb
        self.rb=rb
        self.tensor=tensor
        self.sb_tensor:Optional[torch.Tensor]=None
        self.rb_tensor:Optional[torch.Tensor]=None

    def crt_tensor(self):
        self.sb_tensor=torch.from_numpy(self.sb)
        self.rb_tensor=torch.from_numpy(self.rb)
        self.tensor=True

    def __len__(self):
        return len(self.sb)

    def __getitem__(self, idx):
        if self.tensor:
            return self.sb_tensor[idx].unsqueeze(0),self.rb_tensor[idx].unsqueeze(0)
        else:
            return self.sb[idx],self.rb[idx]

class PreDataset_v2(Dataset):
    '''
    数据集内部采用最大值最小值归一化，网络中无需归一化
    '''
    def __init__(self,t_arr:np.ndarray,sb:np.ndarray,rb:np.ndarray,tensor:bool=False):
        self.t_arr=t_arr
        self.sb=sb
        self.rb=rb
        self.Mn=len(self.sb)
        self.tensor=tensor #标志是否已经tensor化
        self.sb_tensor:Optional[torch.Tensor]=None
        self.sbn_tensor: Optional[torch.Tensor]=None
        self.rb_tensor:Optional[torch.Tensor]=None
        self.rbn_tensor: Optional[torch.Tensor]=None
        #用于反归一化的w和b，是只针对rb的
        self.w: Optional[torch.Tensor]=None
        self.b: Optional[torch.Tensor]=None

    def crt_tensor(self):
        #首先进行min-max normalization
        sb_max: np.ndarray=np.max(self.sb,axis=1).reshape(self.Mn,1)
        sb_min: np.ndarray=np.min(self.sb,axis=1).reshape(self.Mn,1)
        rb_max: np.ndarray=np.max(self.rb,axis=1).reshape(self.Mn,1)
        rb_min: np.ndarray=np.min(self.rb,axis=1).reshape(self.Mn,1)
        sbn=(self.sb-sb_min)/(sb_max-sb_min)
        rbn=(self.rb-rb_min)/(rb_max-rb_min)
        self.sbn_tensor=torch.from_numpy(sbn)
        self.rbn_tensor=torch.from_numpy(rbn)
        #定义用于反归一化的w和b
        self.w=torch.from_numpy(rb_max-rb_min)
        self.b=torch.from_numpy(rb_min)
        #最后创建原信号的tensor
        self.sb_tensor=torch.from_numpy(self.sb)
        self.rb_tensor=torch.from_numpy(self.rb)
        self.tensor=True

    def __len__(self):
        return self.Mn

    def __getitem__(self, idx):
        if self.tensor:
            return self.sbn_tensor[idx].unsqueeze(0),self.rbn_tensor[idx].unsqueeze(0),self.w[idx],self.b[idx]
        else:
            return self.sb[idx],self.rb[idx]

class PreDataset_v3(Dataset):
    '''
    数据集内部采用最大值最小值归一化，网络中无需归一化
    '''
    def __init__(self,t_arr:np.ndarray,sb:np.ndarray,rb:np.ndarray):
        self.t_arr=t_arr
        self.sb=sb
        self.rb=rb
        self.Mn=len(self.sb)
        self.tensored=False #标志是否已经tensor化
        self.sbn_tensor: Optional[torch.Tensor]=None
        self.rbn_tensor: Optional[torch.Tensor]=None

    def crt_tensor(self):
        sbn:np.ndarray=np.empty_like(self.sb)
        rbn:np.ndarray=np.empty_like(self.rb)
        #首先进行min-max normalization
        for i in range(self.Mn):
            sb_max=np.max(self.sb[i])
            sb_min=np.min(self.sb[i])
            sbn[i]=(self.sb[i]-sb_min)/(sb_max-sb_min)
            rb_max=np.max(self.rb[i])
            rb_min=np.min(self.rb[i])
            rbn[i]=(self.rb[i]-rb_min)/(rb_max-rb_min)
        #创建归一化后的tensor
        self.sbn_tensor=torch.from_numpy(sbn)
        self.rbn_tensor=torch.from_numpy(rbn)
        #最后标记已经tensor化
        self.tensored=True

    def __len__(self):
        return self.Mn

    def __getitem__(self, idx):
        if self.tensored:
            return self.sbn_tensor[idx],self.rbn_tensor[idx]
        else:
            return self.sb[idx],self.rb[idx]

class RFFDataset(Dataset):
    def __init__(self,t_arr:np.ndarray,rb:np.ndarray,label:np.ndarray):
        self.t_arr=t_arr
        self.rb=rb
        self.label=label
        self.Mn=len(self.label)
        self.tensored=False  #标志是否已经tensor化
        self.rbn_tensor: Optional[torch.Tensor]=None
        self.label_tensor:Optional[torch.Tensor]=None
        # self.rb1n_tensor:Optional[torch.Tensor]=None

    def crt_tensor(self):
        rbn: np.ndarray=np.empty_like(self.rb)
        #首先进行min-max normalization
        for i in range(self.Mn):
            rb_max=np.max(self.rb[i])
            rb_min=np.min(self.rb[i])
            rbn[i]=(self.rb[i]-rb_min)/(rb_max-rb_min)
        #创建归一化后的tensor
        self.rbn_tensor=torch.from_numpy(rbn)
        self.label_tensor=torch.from_numpy(self.label).long()
        #最后标记已经tensor化
        self.tensored=True

    def __len__(self):
        return self.Mn

    def __getitem__(self, idx):
        if self.tensored:
            return self.rbn_tensor[idx],self.label_tensor[idx]
        else:
            return self.rb[idx],self.label[idx]

class DatasetGnt:
    def __init__(self,T,N,fc,SF,M,beta1,beta2,phi1,phi2,c21,c22,c31,c32):
        self.T=T
        self.N=N
        self.fc=fc
        self.SF=SF
        self.M=M
        self.beta1=beta1
        self.beta2=beta2
        self.phi1=phi1
        self.phi2=phi2
        self.c21=c21
        self.c22=c22
        self.c31=c31
        self.c32=c32
        #定义一个device controller
        self.refresh_dev()
        #定义一个lora signal controller列表
        self.lsc_list:List[Optional[LoraSigCtrl]]=[None]*self.M
        for i in range(self.M):
            lsc=LoraSigCtrl(self.T,self.N,self.fc,self.SF,self.dc.dev_list[i])
            self.lsc_list[i]=lsc

    def refresh_dev(self):
        self.dc = DevCtrl(self.M, self.beta1, self.beta2, self.phi1, self.phi2, self.c21, self.c22, self.c31, self.c32)

    def gnt_pre_dataset(self, n, gamma_s:int, gamma_r:int, pre_dataset_version:str= 'v1'):
        '''
        生成预训练数据集PreDataset并保存

        :param n: preamble的发送次数
        :param gamma_s: 发送端直接测量的SNR
        :param gamma_r: 接收端SNR
        '''
        print('generating pre-dataset...')
        #初始化预训练数据集的s和r
        t_arr=np.linspace(start=0,stop=self.T,num=self.N,endpoint=False)
        s=np.empty(shape=(self.M*n,self.N),dtype=np.complex128)
        r=np.empty(shape=(self.M*n,self.N),dtype=np.complex128)
        pbar=trange(self.M)
        for i in pbar:
            pbar.set_description(f'now processing device {i}')
            #计算当前器件i发送信号的功率ps_i
            rff_arr=self.lsc_list[i].rff_arr
            rff_arr_r,rff_arr_i=rff_arr.real,rff_arr.imag #取实部和虚部
            ps_i=1/self.N*(np.inner(rff_arr_r,rff_arr_r)+np.inner(rff_arr_i,rff_arr_i))
            # 根据输入信噪比计算噪声功率
            pn_s=ps_i*math.pow(10,-gamma_s/10)
            pn_r=ps_i*math.pow(10,-gamma_r/10)
            for j in range(n):
                #计算当前数据在数据集中的索引
                idx=i*self.M+j
                s[idx]=rff_arr+self.gnt_gaussian_noise(self.N,pn_s)
                r[idx]=rff_arr+self.gnt_gaussian_noise(self.N,pn_r)
        #下面对带通信号进行理想解调得到基带信号
        print('demodulating...')
        demodulator = np.exp(1j*2*pi*self.fc*t_arr)
        sb = np.empty(shape=(self.M * n, self.N), dtype=np.float64)
        rb = np.empty(shape=(self.M * n, self.N), dtype=np.float64)
        # 将解调后的信号取实部
        for i in trange(self.M*n):
            sb[i]=(s[i]*demodulator).real
            rb[i]=(r[i]*demodulator).real
        pre_dataset = PreDataset(t_arr,sb,rb)
        print('pre-dataset created')
        print('saving pre-dataset...')
        pre_dataset_root = '../dataset'
        filename = f'pre_dataset_{pre_dataset_version}.pkl'
        path=os.path.abspath(os.path.join(pre_dataset_root, filename))
        fw = open(path, 'wb')
        pkl.dump(pre_dataset, fw)
        fw.close()
        print(f'pre-dataset saved at {path}')

    def gnt_pre_dataset_v2(self, n, gamma_s:int, gamma_r:int, pre_dataset_version:str= 'v1'):
        '''
        生成预训练数据集PreDataset_v2并保存

        :param n: preamble的发送次数
        :param gamma_s: 发送端直接测量的SNR
        :param gamma_r: 接收端SNR
        '''
        print('generating pre-dataset...')
        #初始化预训练数据集的s和r
        t_arr=np.linspace(start=0,stop=self.T,num=self.N,endpoint=False)
        s=np.empty(shape=(self.M*n,self.N),dtype=np.complex128)
        r=np.empty(shape=(self.M*n,self.N),dtype=np.complex128)
        pbar=trange(self.M)
        for i in pbar:
            pbar.set_description(f'now processing device {i}')
            #计算当前器件i发送信号的功率ps_i
            rff_arr=self.lsc_list[i].rff_arr
            rff_arr_r,rff_arr_i=rff_arr.real,rff_arr.imag #取实部和虚部
            ps_i=1/self.N*(np.inner(rff_arr_r,rff_arr_r)+np.inner(rff_arr_i,rff_arr_i))
            # 根据输入信噪比计算噪声功率
            pn_s=ps_i*math.pow(10,-gamma_s/10)
            pn_r=ps_i*math.pow(10,-gamma_r/10)
            for j in range(n):
                #计算当前数据在数据集中的索引
                idx=i*n+j
                s[idx]=rff_arr+self.gnt_gaussian_noise(self.N,pn_s)
                r[idx]=rff_arr+self.gnt_gaussian_noise(self.N,pn_r)
        #下面对带通信号进行理想解调得到基带信号
        print('demodulating...')
        demodulator = np.exp(1j*2*pi*self.fc*t_arr)
        sb = np.empty(shape=(self.M * n, self.N), dtype=np.float64)
        rb = np.empty(shape=(self.M * n, self.N), dtype=np.float64)
        # 将解调后的信号取实部
        for i in trange(self.M*n):
            sb[i]=(s[i]*demodulator).real
            rb[i]=(r[i]*demodulator).real
        pre_dataset = PreDataset_v2(t_arr,sb,rb)
        print('pre-dataset created')
        print('saving pre-dataset...')
        pre_dataset_root = '../dataset'
        filename = f'pre_dataset_{pre_dataset_version}.pkl'
        path=os.path.abspath(os.path.join(pre_dataset_root, filename))
        fw = open(path, 'wb')
        pkl.dump(pre_dataset, fw)
        fw.close()
        print(f'pre-dataset saved at {path}')

    def gnt_pre_dataset_v3(self, n, gamma_s:int, gamma_r:int, pre_dataset_version:str= 'v1'):
        '''
        生成预训练数据集PreDataset_v3并保存

        :param n: preamble的发送次数
        :param gamma_s: 发送端直接测量的SNR
        :param gamma_r: 接收端SNR
        '''
        print('generating pre-dataset...')
        #初始化预训练数据集的s和r
        t_arr=np.linspace(start=0,stop=self.T,num=self.N,endpoint=False)
        s=np.empty(shape=(self.M*n,self.N),dtype=np.complex128)
        r=np.empty(shape=(self.M*n,self.N),dtype=np.complex128)
        pbar=trange(self.M)
        for i in pbar:
            pbar.set_description(f'now processing device {i}')
            #计算当前器件i发送信号的功率ps_i
            rff_arr=self.lsc_list[i].rff_arr
            rff_arr_r,rff_arr_i=rff_arr.real,rff_arr.imag #取实部和虚部
            ps_i=1/self.N*(np.inner(rff_arr_r,rff_arr_r)+np.inner(rff_arr_i,rff_arr_i))
            # 根据输入信噪比计算噪声功率
            pn_s=ps_i*math.pow(10,-gamma_s/10)
            pn_r=ps_i*math.pow(10,-gamma_r/10)
            for j in range(n):
                #计算当前数据在数据集中的索引
                idx=i*n+j
                s[idx]=rff_arr+self.gnt_gaussian_noise(self.N,pn_s)
                r[idx]=rff_arr+self.gnt_gaussian_noise(self.N,pn_r)
        #下面对带通信号进行理想解调得到基带信号
        print('demodulating...')
        demodulator = np.exp(1j*2*pi*self.fc*t_arr)
        #sbc和rbc为复信号数组
        sbc:np.ndarray=s*demodulator
        rbc:np.ndarray=r*demodulator
        Mn=self.M*n
        sb = np.empty(shape=(Mn,2,self.N), dtype=np.float64)
        rb = np.empty(shape=(Mn,2,self.N), dtype=np.float64)
        for i in range(Mn):
            sb[i][0]=sbc[i].real
            sb[i][1]=sbc[i].imag
            rb[i][0]=rbc[i].real
            rb[i][1]=rbc[i].imag
        pre_dataset:PreDataset_v3=PreDataset_v3(t_arr,sb,rb)
        print('pre-dataset created')
        print('saving pre-dataset...')
        pre_dataset_root = '../dataset'
        filename = f'pre_dataset_{pre_dataset_version}.pkl'
        path=os.path.abspath(os.path.join(pre_dataset_root, filename))
        fw = open(path, 'wb')
        pkl.dump(pre_dataset, fw)
        fw.close()
        print(f'pre-dataset saved at {path}')

    def gnt_rff_dataset(self, n:int, gamma_r:int, rff_dataset_version:str= 'v1'):
        '''
        生成RFF数据集并保存

        :param n: preamble的发送次数
        :param gamma_r: 接收端SNR
        '''
        print('generating rff dataset...')
        #更新器件
        self.refresh_dev()
        #初始化RFF数据集的r和label
        t_arr=np.linspace(start=0,stop=self.T,num=self.N,endpoint=False)
        r=np.empty(shape=(self.M*n,self.N),dtype=np.complex128)
        label=np.empty(self.M*n,dtype=int)
        #建立RFF数据集
        pbar=trange(self.M)
        for i in pbar:
            pbar.set_description(f'now processing device {i}')
            #计算当前器件i发送信号的功率ps_i
            rff_arr=self.lsc_list[i].rff_arr
            rff_arr_r, rff_arr_i = rff_arr.real, rff_arr.imag
            ps_i=1/self.N*(np.inner(rff_arr_r,rff_arr_r)+np.inner(rff_arr_i,rff_arr_i))
            # 根据输入信噪比计算噪声功率
            pn_r=ps_i*math.pow(10,-gamma_r/10)
            for j in range(n):
                #计算当前数据在数据集中的索引
                idx=i*n+j
                r[idx]=rff_arr+self.gnt_gaussian_noise(self.N,pn_r)
                label[idx]=int(i)
        #下面对带通信号进行理想解调得到基带信号
        print('demodulating...')
        demodulator=np.exp(1j*2*pi*self.fc*t_arr)
        #rbc为复信号数组
        rbc:np.ndarray=r*demodulator
        Mn=self.M*n
        rb=np.empty(shape=(Mn,2,self.N),dtype=np.float64)
        for i in range(Mn):
            rb[i][0]=rbc[i].real
            rb[i][1]=rbc[i].imag
        rff_dataset:RFFDataset=RFFDataset(t_arr,rb,label)
        rff_dataset_root='../dataset'
        filename=f'rff_dataset_{rff_dataset_version}.pkl'
        path=os.path.abspath(os.path.join(rff_dataset_root,filename))
        fw=open(path,'wb')
        pkl.dump(rff_dataset,fw)
        fw.close()
        print(f'rff dataset saved at {path}')


    @staticmethod
    def gnt_gaussian_noise(N,pn):
        '''
        生成高斯噪声

        :param N: 信号点数
        :param pn: 噪声功率
        :return: N点高斯噪声
        '''
        return np.random.normal(0,math.sqrt(pn/2),N)+1j*np.random.normal(0,math.sqrt(pn/2),N)