import matplotlib.pyplot as plt
from .dataset_gnt import *
import os
import pickle as pkl

def read_dataset(mode:int=0,dataset_version:str='v1'):
    '''
    :param mode: 0表示PreDataset，1表示RFFDataset
    :param dataset_version: 数据集版本
    :return: dataset
    '''
    dataset_type='pre_dataset' if mode==0 else 'rff_dataset'
    path=os.path.abspath(f'../dataset/{dataset_type}_{dataset_version}.pkl')
    fr=open(path,'rb')
    if mode==0:
        dataset:PreDataset_v3=pkl.load(fr)
        return dataset
    else:
        dataset:RFFDataset=pkl.load(fr)
        return dataset


def display(mode:int=0,dataset_version:str='v1'):
    pre_dataset=read_dataset(mode=mode,dataset_version=dataset_version)
    #选取单个一维数组作为返回值用于绘图
    t_arr,sb,rb=pre_dataset.t_arr,pre_dataset.sb[-1],pre_dataset.rb[-1]
    l=int(len(t_arr)*0.1)
    t_arr,sb,rb=t_arr[:l],sb[:l],rb[:l]
    plt.subplot(2,1,1)
    plt.plot(t_arr,sb)
    plt.title('sent baseband')
    plt.subplot(2,1,2)
    plt.plot(t_arr,rb)
    plt.title('received baseband')
    plt.show()
