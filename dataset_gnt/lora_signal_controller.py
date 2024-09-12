import numpy as np
import math
from .signal_controller_base import SigCtrlBase

pi=math.pi

class LoraSigCtrl(SigCtrlBase):
    def __init__(self,T,N,fc,SF,device,init_mode:int=1):
        super().__init__(T, N, fc)
        if init_mode==1:
            self.init1(SF,device)
        elif init_mode==2:
            self.init2(SF,device)

    def init1(self,SF,device):
        '''
        构造函数1，使用带通信号制作数据集，不解调
        '''
        self.SF = SF
        self.set_baseband_param()
        self.cal_baseband()
        self.device = device
        self.cal_bandpass()

    def init2(self,SF,device):
        '''
        构造函数2，解调得到基带信号制作数据集
        '''
        self.init1(SF,device)

    def set_baseband_param(self):
        '''
        定义Lora基带信号所需参数
        '''
        self.B=(2**self.SF)/self.T

    def cal_baseband(self):
        '''
        计算Lora基带信号a_arr
        '''
        self.a_arr=np.exp(1j*(-pi*self.B*self.t_arr+pi*self.B/self.T*self.t_arr**2))

    def cal_bandpass(self):
        '''
        计算Lora带通信号preamble序列rff_arr
        '''
        #首先计算IQ失衡
        mod_arr=np.cos(2*pi*self.fc*self.t_arr)+1j*self.device.beta*np.sin(2*pi*self.fc*self.t_arr+self.device.phi)
        #计算IQ调制后的信号sm
        sm=self.a_arr*mod_arr
        #计算非线性放大后的信号
        sm_abs=np.abs(sm)
        self.rff_arr=(1+self.device.c2*sm_abs+self.device.c3*np.square(sm_abs))*sm