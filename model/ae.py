import torch
import torch.nn as nn

class iSigmoid(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,x):
        return torch.log(x/(1-x))

class SoftClamp(nn.Module):
    def __init__(self, epsilon=1e-6):
        super().__init__()
        self.epsilon = epsilon  # 设置略大于0或略小于1的偏移量

    def forward(self, x):
        # torch.where 实现条件分支处理
        x = torch.where(x <= 0, torch.full_like(x, self.epsilon), x)  # 小于等于0的输出略大于0
        x = torch.where(x >= 1, torch.full_like(x, 1 - self.epsilon), x)  # 大于等于1的输出略小于1
        return x

class HardClamp(nn.Module):
    def __init__(self):
        super(HardClamp,self).__init__()

    def forward(self, x):
        return torch.clamp(x, 0, 1)

class InConv1d(nn.Module):
    def __init__(self,in_ch=2,out_ch=64,ker=15):
        super().__init__()
        pad=int((ker-1)/2)
        self.net=nn.Sequential(
            nn.Sigmoid(),
            nn.Conv1d(in_channels=in_ch,out_channels=out_ch,kernel_size=ker,
                      padding=pad,stride=1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU()
        )

    def forward(self,X):
        return self.net(X)

class InConv1d_v2(nn.Module):
    def __init__(self,in_ch=2,out_ch=64,ker=15):
        super().__init__()
        pad=int((ker-1)/2)
        self.net=nn.Sequential(
            nn.Conv1d(in_channels=in_ch,out_channels=out_ch,kernel_size=ker,
                      padding=pad,stride=1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU()
        )

    def forward(self,X):
        return self.net(X)

class InConv2d(nn.Module):
    def __init__(self,in_ch=1,out_ch=64,ker:tuple=(2,31)):
        super().__init__()
        pad:tuple=(0,(ker[1]-1)//2)
        self.net=nn.Sequential(
            nn.Conv2d(in_channels=in_ch,out_channels=out_ch,kernel_size=ker,
                      padding=pad,stride=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self,X):
        return self.net(X)

class Encoder1d(nn.Module):
    def __init__(self,in_ch,out_ch,ker=15):
        super().__init__()
        pad=int((ker-1)/2)
        self.net=nn.Sequential(
            nn.MaxPool1d(kernel_size=ker,padding=pad,stride=2),
            nn.Conv1d(in_channels=in_ch,out_channels=2*in_ch,kernel_size=ker,
                      padding=pad,stride=1),
            nn.BatchNorm1d(2*in_ch),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=ker,padding=pad,stride=2),
            nn.Conv1d(in_channels=2*in_ch,out_channels=out_ch,kernel_size=ker,
                      padding=pad,stride=1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU()
        )

    def forward(self,X):
        return self.net(X)

class Encoder2d(nn.Module):
    def __init__(self,in_ch,out_ch,ker:tuple=(2,31)):
        super().__init__()
        pad:tuple=(0,(ker[1]-1)//2)
        self.net=nn.Sequential(
            nn.MaxPool2d(kernel_size=ker,padding=pad,stride=2),
            nn.Conv2d(in_channels=in_ch,out_channels=2*in_ch,kernel_size=ker,
                      padding=pad,stride=1),
            nn.BatchNorm2d(2*in_ch),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=ker,padding=pad,stride=2),
            nn.Conv2d(in_channels=2*in_ch,out_channels=out_ch,kernel_size=ker,
                      padding=pad,stride=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self,X):
        return self.net(X)

class Decoder1d(nn.Module):
    def __init__(self,in_ch,out_ch,ker=15):
        super().__init__()
        pad=int((ker-1)/2)
        self.net=nn.Sequential(
            nn.Upsample(scale_factor=2,mode='linear',align_corners=True),
            nn.Conv1d(in_channels=in_ch,out_channels=in_ch//2,kernel_size=ker,
                      padding=pad,stride=1),
            nn.BatchNorm1d(in_ch//2),
            nn.ReLU(),
            nn.Upsample(scale_factor=2,mode='linear',align_corners=True),
            nn.Conv1d(in_channels=in_ch//2,out_channels=out_ch,kernel_size=ker,
                      padding=pad,stride=1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU()
        )

    def forward(self,X):
        return self.net(X)

class Decoder2d(nn.Module):
    def __init__(self,in_ch,out_ch,ker:tuple=(2,31)):
        super().__init__()
        pad:tuple=(0,(ker[1]-1)//2)
        self.net=nn.Sequential(
            nn.Upsample(scale_factor=2,mode='linear',align_corners=True),
            nn.Conv2d(in_channels=in_ch,out_channels=in_ch//2,kernel_size=ker,
                      padding=pad,stride=1),
            nn.BatchNorm2d(in_ch//2),
            nn.ReLU(),
            nn.Upsample(scale_factor=2,mode='linear',align_corners=True),
            nn.Conv2d(in_channels=in_ch//2,out_channels=out_ch,kernel_size=ker,
                      padding=pad,stride=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self,X):
        return self.net(X)

class OutConv1d(nn.Module):
    def __init__(self,in_ch=64,out_ch=2,ker=15):
        super().__init__()
        pad=int((ker-1)/2)
        self.net=nn.Sequential(
            nn.Conv1d(in_channels=in_ch,out_channels=out_ch,kernel_size=ker,
                      padding=pad,stride=1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
            SoftClamp(),
            iSigmoid()
        )

    def forward(self,X):
        return self.net(X)

class OutConv1d_v2(nn.Module):
    def __init__(self,in_ch=64,out_ch=2,ker=15):
        super().__init__()
        pad=int((ker-1)/2)
        self.net=nn.Sequential(
            nn.Conv1d(in_channels=in_ch,out_channels=out_ch,kernel_size=ker,
                      padding=pad,stride=1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
            HardClamp()
        )

    def forward(self,X):
        return self.net(X)

class OutConv1d_v3(nn.Module):
    def __init__(self,in_ch=64,out_ch=2,ker=15):
        super().__init__()
        pad=int((ker-1)/2)
        self.net=nn.Sequential(
            nn.Conv1d(in_channels=in_ch,out_channels=out_ch,kernel_size=ker,
                      padding=pad,stride=1),
            nn.BatchNorm1d(out_ch)
        )

    def forward(self,X):
        return self.net(X)

class OutConv2d(nn.Module):
    def __init__(self,in_ch=64,out_ch=1,ker:tuple=(2,31)):
        super().__init__()
        pad:tuple=(0,(ker[1]-1)//2)
        self.net=nn.Sequential(
            nn.Conv2d(in_channels=in_ch,out_channels=out_ch,kernel_size=ker,
                      padding=pad,stride=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self,X):
        return self.net(X)

class AE1d(nn.Module):
    def __init__(self,ch=64,ker=15):
        '''
        :param ch: 编码器的输入通道数，即输入卷积的输出通道数
        :param ker: 一维卷积核大小
        '''
        super().__init__()
        pad=int((ker-1)/2)
        self.net=nn.Sequential(
            InConv1d(in_ch=1,out_ch=ch,ker=ker),
            Encoder1d(in_ch=ch,out_ch=ch*4,ker=ker),
            Decoder1d(in_ch=ch*4,out_ch=ch,ker=ker),
            OutConv1d(in_ch=ch,out_ch=1,ker=ker)
        )

    def forward(self,X):
        return self.net(X)

class AE1d_v2(nn.Module):
    def __init__(self,ch=64,ker=15):
        '''
        :param ch: 编码器的输入通道数，即输入卷积的输出通道数
        :param ker: 一维卷积核大小
        '''
        super().__init__()
        pad=int((ker-1)/2)
        self.net=nn.Sequential(
            InConv1d_v2(in_ch=1,out_ch=ch,ker=ker),
            Encoder1d(in_ch=ch,out_ch=ch*4,ker=ker),
            Decoder1d(in_ch=ch*4,out_ch=ch,ker=ker),
            OutConv1d_v2(in_ch=ch,out_ch=1,ker=ker)
        )

    def forward(self,X):
        return self.net(X)

class AE1d_v3(nn.Module):
    def __init__(self,ch=64,ker=15):
        '''
        :param ch: 编码器的输入通道数，即输入卷积的输出通道数
        :param ker: 一维卷积核大小
        '''
        super().__init__()
        pad=int((ker-1)/2)
        self.net=nn.Sequential(
            InConv1d_v2(in_ch=2,out_ch=ch,ker=ker),
            Encoder1d(in_ch=ch,out_ch=ch*4,ker=ker),
            Decoder1d(in_ch=ch*4,out_ch=ch,ker=ker),
            OutConv1d_v3(in_ch=ch,out_ch=2,ker=ker)
        )

    def forward(self,X):
        return self.net(X)

class AE2d(nn.Module):
    def __init__(self,ch:int=64,ker:tuple=(2,31)):
        '''
        :param ch: 编码器的输入通道数，即输入卷积的输出通道数
        :param ker: 一维卷积核大小
        '''
        super().__init__()
        self.net=nn.Sequential(
            InConv2d(in_ch=1,out_ch=ch,ker=ker),
            Encoder2d(in_ch=ch,out_ch=ch*4,ker=ker),
            Decoder2d(in_ch=ch*4,out_ch=ch,ker=ker),
            OutConv2d(in_ch=ch,out_ch=1,ker=ker)
        )

    def forward(self,X):
        return self.net(X)
