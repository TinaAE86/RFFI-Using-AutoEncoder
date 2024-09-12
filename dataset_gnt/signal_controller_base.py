import numpy as np
import math

pi=math.pi

class SigCtrlBase:
    def __init__(self,T,N,fc):
        '''
        :param T: 信号持续时间
        :param N: 信号采样点数
        :param fc: 载波频率
        '''
        self.T=T
        self.N=int(N)
        self.fc=fc
        self.Ns=T/N #采样间隔
        self.t_arr=np.linspace(start=0,stop=T,num=N,endpoint=False)
        self.a_arr=None #基带信号序列
        self.rff_arr=None #含有射频指纹的信号序列，即实际发送信号

    def set_baseband_param(self):
        '''
        （基类）定义基带信号所需的参数
        '''
        pass

    def cal_baseband(self):
        '''
        (基类）计算基带信号a_arr
        '''
        pass