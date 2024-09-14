import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Optional
from dataset_gnt.dataset_gnt import *
import os
from torch.utils.data import DataLoader
from model.ae import *

class Train():
    def __init__(self,rff_dataset_version:str='v1',ae_version:str='v1',encode_channels:int=64,
                 ae_kernel_size:int=31,batch_size:int=4,lr:float=1e-5,epoch_num:int=100,
                 plot_ae:bool=False,train:bool=False):
        self.rff_dataset:Optional[RFFDataset]=None
        #rff_dataset1是经过ae后并重新归一化的数据集
        self.rff_dataset1:Optional[RFFDataset]=None
        self.batch_size=batch_size
        self.ae:AE1d_v3=AE1d_v3(ch=encode_channels,ker=ae_kernel_size).double()
        self.ae_version=ae_version
        self.cpu=torch.device('cpu')
        self.device=torch.device('cuda')
        self.read_rff_dataset(rff_dataset_version=rff_dataset_version)
        self.dataloader:Optional[DataLoader]=None
        self.load_ae()
        self.crt_dataloader()
        self.loss_func=nn.CrossEntropyLoss()
        self.loss_list: list=[None]*epoch_num
        self.epoch_list: list=[i for i in range(epoch_num)]
        #定义特征提取分类网络模型

        if plot_ae:
            self.plot_ae_curve()
        if train:
            self.train(epoch_num=epoch_num)

    def read_rff_dataset(self,rff_dataset_version):
        print('reading rff dataset...')
        path=os.path.abspath(f'../dataset/rff_dataset_{rff_dataset_version}.pkl')
        fr=open(path,'rb')
        self.rff_dataset=pkl.load(fr)
        print('rff dataset loaded')

    def load_ae(self):
        print('loading ae parameters...')
        path=os.path.abspath(f'../model_param/ae_{self.ae_version}.pt')
        self.ae.load_state_dict(torch.load(path,weights_only=True))
        self.ae.eval()
        print('ae parameters loaded')

    def crt_dataloader(self):
        print('creating dataloader...')
        #根据rff dataset中numpy数组创建tensor
        self.rff_dataset.crt_tensor()
        #将ae加载至GPU
        self.ae=self.ae.to(self.device)
        rb1:np.ndarray=np.empty_like(self.rff_dataset.rb)
        label1:np.ndarray=np.empty_like(self.rff_dataset.label)
        with torch.no_grad():
            for i in range(self.rff_dataset.Mn):
                rb_tensor,label_tensor=self.rff_dataset[i]
                rb_tensor:torch.Tensor=rb_tensor.unsqueeze(0).to(self.device)
                output:torch.Tensor=self.ae(rb_tensor).squeeze(0).detach().to(self.cpu)
                rb1_i:np.ndarray=output.numpy()
                rb1[i]=rb1_i
                label1[i]=label_tensor.numpy()
                # del rb_tensor
        #将ae从GPU移除以减少开销
        self.ae=self.ae.to(self.cpu)
        torch.cuda.empty_cache()
        #用降噪后的数据建立rff_dataset1
        self.rff_dataset1=RFFDataset(t_arr=self.rff_dataset.t_arr,rb=rb1,label=label1)
        #归一化操作在tensor化步骤中
        self.rff_dataset1.crt_tensor()
        self.dataloader=DataLoader(dataset=self.rff_dataset1,shuffle=True,batch_size=self.batch_size,
                                   num_workers=4,drop_last=True)
        print('dataloader created')

    def plot_ae_curve(self):
        t_arr:np.ndarray=self.rff_dataset1.t_arr
        rbn:np.ndarray=self.rff_dataset.rbn_tensor[-1].numpy()
        rbn1:np.ndarray=self.rff_dataset1.rbn_tensor[-1].numpy()
        plt.subplot(2,2,1)
        plt.plot(t_arr,rbn[0])
        plt.title('rbn I branch')
        plt.subplot(2,2,2)
        plt.plot(t_arr,rbn[1])
        plt.title('rbn Q branch')
        plt.subplot(2,2,3)
        plt.plot(t_arr,rbn1[0])
        plt.title('rbn1 I branch')
        plt.subplot(2,2,4)
        plt.plot(t_arr,rbn1[1])
        plt.title('rbn1 Q branch')
        plt.show()

    def train(self,epoch_num):
        pass



