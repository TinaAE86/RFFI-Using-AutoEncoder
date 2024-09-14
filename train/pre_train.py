import torch
import torch.nn as nn
from dataset_gnt.dataset_gnt import *
from torch.utils.data import Dataset,DataLoader
import os
import pickle as pkl
import numpy as np
from typing import List,Optional
from model.ae import *
from tqdm import tqdm,trange
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import time

class PreTrain():
    def __init__(self,batch_size:int=4,pre_dataset_version:str='v1',epoch_num:int=100,
                 encode_channels:int=64,kernel_size:int|tuple=31,lr:float=1e-5,
                 ae_version:str='v1',train:bool=False):
        '''
        可能需要修改的是self.pre_dataset的类版本、self.model的类版本、self.train的版本
        '''
        self.pre_dataset:Optional[PreDataset_v3]=None
        self.batch_size=batch_size
        self.dataloader:Optional[DataLoader]=None
        self.ae_version=ae_version
        self.read_pre_dataset(pre_dataset_version=pre_dataset_version)
        self.crt_dataloader()
        self.device=torch.device('cuda')
        self.loss_func=nn.MSELoss()
        self.loss_list:list=[None]*epoch_num
        self.epoch_list:list=[i for i in range(epoch_num)]
        #定义网络模型，并指定数据精度为float64
        self.model=AE1d_v3(ch=encode_channels,ker=kernel_size).double()
        #将网络模型加载至GPU
        self.model=self.model.to(self.device)
        #定义优化器
        self.optim=torch.optim.Adam(self.model.parameters(),lr=lr)
        #训练网络
        if train:
            self.train_v3(epoch_num)

    def read_pre_dataset(self,pre_dataset_version:str='v1'):
        print('reading pre-dataset...')
        path=os.path.abspath(f'../dataset/pre_dataset_{pre_dataset_version}.pkl')
        fr=open(path,'rb')
        self.pre_dataset=pkl.load(fr)
        print('pre-dataset loaded')

    @staticmethod
    def sig2tensor_v1(sig:np.ndarray):
        '''
        将numpy数组形式的sig分离实部和虚部拼成双通道tensor

        :param sig: N点复信号
        :return: tensor
        '''
        sig_real,sig_imag=torch.from_numpy(sig.real),torch.from_numpy(sig.imag)
        return torch.stack((sig_real,sig_imag))

    @staticmethod
    def sig2tensor_v2(sig:np.ndarray):
        '''
        将numpy数组形式的sig取实部并转化为tensor

        :param sig: N点复信号
        :return: tensor
        '''
        return torch.from_numpy(sig.real)

    # def crt_dataloader(self):
    #     print('creating dataloader...')
    #     t_arr:np.ndarray=self.pre_dataset.t_arr
    #     s:np.ndarray=self.pre_dataset.s
    #     r:np.ndarray=self.pre_dataset.r
    #     l=len(s)
    #     sig_list:List[Optional[List[torch.Tensor]]]=[None]*l
    #     for i in range(l):
    #         sig_list[i]=[self.sig2tensor_v1(s[i]), self.sig2tensor_v1(r[i])]
    #     self.pre_dataset_tensor=PreDatasetTensor_v1(sig_list)
    #     self.dataloader=DataLoader(self.pre_dataset_tensor, shuffle=True, batch_size=self.batch_size,
    #                                num_workers=4, drop_last=True)
    #     print('dataloader created')

    def crt_dataloader(self):
        print('creating dataloader...')
        #根据pre-dataset中numpy数组创建tensor
        self.pre_dataset.crt_tensor()
        self.dataloader=DataLoader(self.pre_dataset,shuffle=True,batch_size=self.batch_size,
                                   num_workers=4,drop_last=True)
        print('dataloader created')


    def train(self,epoch_num:int):
        loss:list=[0.0] #记录loss进行累加
        batch_num:list=[0] #记录batch数量
        #增加一个空白对照，记录初始sb_tensor和rb_tensor的均方误差
        mse_func=nn.MSELoss()
        mse:list=[0.0]
        print('now training...')
        time.sleep(0.1)
        writer=SummaryWriter(os.path.abspath(f'../runs/pre_train/ae_{self.ae_version}'))
        pbar=trange(epoch_num)
        for epoch in pbar:
            loss[0]=0.0 #将loss归零
            batch_num[0]=0 #将batch_num归零
            mse[0]=0.0 #将空白对照组mse归零
            for sb_tensor,rb_tensor in self.dataloader:
                sb_tensor,rb_tensor=sb_tensor.to(self.device),rb_tensor.to(self.device)
                self.optim.zero_grad()
                output=self.model(rb_tensor)
                loss_t:torch.Tensor=self.loss_func(output,sb_tensor)
                loss[0]+=loss_t.item()
                batch_num[0]+=1
                #记录空白对照
                original_mse_tensor:torch.Tensor=mse_func(sb_tensor,rb_tensor)
                mse[0]+=original_mse_tensor.item()
                #反向传播
                loss_t.backward()
                #更新参数
                self.optim.step()
            loss[0]/=batch_num[0]
            mse[0]/=batch_num[0]
            self.loss_list[epoch]=loss[0]
            writer.add_scalar('pre_train/loss',loss[0],epoch)
            writer.add_scalar('pre_train/origin MSE',mse[0],epoch)
            pbar.set_description(f'epoch {epoch}, loss={loss[0]}, origin MSE={mse[0]}')
        print('training finished')
        writer.close()
        self.save_model()
        self.plot_loss_curve()

    def train_v2(self,epoch_num:int):
        loss:list=[0.0] #记录loss进行累加
        batch_num:list=[0] #记录batch数量
        #增加一个空白对照，记录初始sb_tensor和rb_tensor的均方误差
        mse_func=nn.MSELoss()
        mse:list=[0.0]
        print('now training...')
        time.sleep(0.1)
        writer=SummaryWriter(os.path.abspath(f'../runs/pre_train/ae_{self.ae_version}'))
        pbar=trange(epoch_num)
        for epoch in pbar:
            loss[0]=0.0 #将loss归零
            batch_num[0]=0 #将batch_num归零
            mse[0]=0.0 #将空白对照组mse归零
            for sbn_tensor,rbn_tensor,w,b in self.dataloader:
                sbn_tensor,rbn_tensor=sbn_tensor.to(self.device),rbn_tensor.to(self.device)
                self.optim.zero_grad()
                output=self.model(rbn_tensor)
                loss_t:torch.Tensor=self.loss_func(output,sbn_tensor)
                loss[0]+=loss_t.item()
                batch_num[0]+=1
                #记录空白对照
                original_mse_tensor:torch.Tensor=mse_func(sbn_tensor,rbn_tensor)
                mse[0]+=original_mse_tensor.item()
                #反向传播
                loss_t.backward()
                #更新参数
                self.optim.step()
            loss[0]/=batch_num[0]
            mse[0]/=batch_num[0]
            self.loss_list[epoch]=loss[0]
            writer.add_scalar('pre_train/loss',loss[0],epoch)
            writer.add_scalar('pre_train/original MSE',mse[0],epoch)
            pbar.set_description(f'epoch {epoch}, loss={loss[0]}, original MSE={mse[0]}')
        print('training finished')
        writer.close()
        self.save_model()
        self.save_res()
        self.plot_loss_curve()

    def train_v3(self,epoch_num:int):
        loss:list=[0.0] #记录loss进行累加
        batch_num:list=[0] #记录batch数量
        #增加一个空白对照，记录初始sb_tensor和rb_tensor的均方误差
        mse_func=nn.MSELoss()
        mse:list=[0.0]
        print('now training...')
        time.sleep(0.1)
        writer=SummaryWriter(os.path.abspath(f'../runs/pre_train/ae_{self.ae_version}'))
        pbar=trange(epoch_num)
        for epoch in pbar:
            loss[0]=0.0 #将loss归零
            batch_num[0]=0 #将batch_num归零
            mse[0]=0.0 #将空白对照组mse归零
            for sbn_tensor,rbn_tensor in self.dataloader:
                sbn_tensor,rbn_tensor=sbn_tensor.to(self.device),rbn_tensor.to(self.device)
                self.optim.zero_grad()
                output=self.model(rbn_tensor)
                loss_t:torch.Tensor=self.loss_func(output,sbn_tensor)
                loss[0]+=loss_t.item()
                batch_num[0]+=1
                #记录空白对照
                original_mse_tensor:torch.Tensor=mse_func(sbn_tensor,rbn_tensor)
                mse[0]+=original_mse_tensor.item()
                #反向传播
                loss_t.backward()
                #更新参数
                self.optim.step()
            loss[0]/=batch_num[0]
            mse[0]/=batch_num[0]
            self.loss_list[epoch]=loss[0]
            writer.add_scalar('pre_train/loss',loss[0],epoch)
            writer.add_scalar('pre_train/original MSE',mse[0],epoch)
            pbar.set_description(f'epoch {epoch}, loss={loss[0]}, original MSE={mse[0]}')
        print('training finished')
        writer.close()
        self.save_model()
        self.save_res()
        self.plot_loss_curve()

    def save_model(self):
        print('saving model parameters...')
        root='../model_param'
        fn=f'ae_{self.ae_version}.pt'
        path=os.path.abspath(os.path.join(root,fn))
        torch.save(self.model.state_dict(),path)
        print(f'model parameters saved at {path}')

    def save_res(self):
        print('saving training results...')
        root='../pkl/pre_train_res'
        fn=f'ae_{self.ae_version}_res.pkl'
        path=os.path.abspath(os.path.join(root,fn))
        fw=open(path,'wb')
        obj=(self.epoch_list,self.loss_list)
        pkl.dump(obj,fw)
        fw.close()
        print(f'training results saved at {path}')

    def plot_loss_curve(self):
        plt.plot(self.epoch_list,self.loss_list)
        plt.show()
