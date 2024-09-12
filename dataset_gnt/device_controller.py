import numpy as np

class Device:
    '''
    模拟器件，含有自身的参数
    '''
    def __init__(self,beta,phi,c2,c3):
        self.beta=beta
        self.phi=phi
        self.c2=c2
        self.c3=c3

class DevCtrl:
    def __init__(self,M,beta1,beta2,phi1,phi2,c21,c22,c31,c32):
        self.M=M # 器件总数
        self.beta1=beta1
        self.beta2=beta2
        self.phi1=phi1
        self.phi2=phi2
        self.c21=c21
        self.c22=c22
        self.c31=c31
        self.c32=c32
        self.dev_list=[None]*self.M
        self.set_dev()

    def set_dev(self):
        '''
        生成M个器件
        '''
        for i in range(self.M):
            beta=np.random.uniform(self.beta1,self.beta2)
            phi=np.random.uniform(self.phi1,self.phi2)
            c2=np.random.uniform(self.c21,self.c22)
            c3=np.random.uniform(self.c31,self.c32)
            self.dev_list[i]=Device(beta,phi,c2,c3)