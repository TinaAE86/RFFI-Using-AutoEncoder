import torch

from train.pre_train import *
from dataset_gnt.display import *

def main1():
    '''
    预训练并保存预训练网络
    '''
    pre_train_obj=PreTrain(batch_size=8,pre_dataset_version='v4',epoch_num=100,encode_channels=4,kernel_size=15,
                           lr=6e-6,ae_version='v12',train=True)

def main2():
    '''
    绘制使用预训练网络降噪后的信号的图像
    '''
    #声明一个模型
    model=AE1d_v2(ch=2,ker=51).double()
    #加载模型参数
    print('loading model parameters...')
    path=os.path.abspath('../model_param/ae_v9.pt')
    model.load_state_dict(torch.load(path,weights_only=True))
    model.eval()
    print('model parameters loaded')
    pre_dataset=read_dataset(mode=0,dataset_version='v3')
    pre_dataset.crt_tensor()
    t_arr,sbn,rbn_tensor=pre_dataset.t_arr,pre_dataset.sbn_tensor[-1].detach().numpy(),\
                         pre_dataset.rbn_tensor[-1].unsqueeze(0).unsqueeze(0)
    l = int(len(t_arr))
    output=model(rbn_tensor).squeeze(0).squeeze(0).detach().numpy()
    t_arr, sbn, rbn,output = t_arr[:l], sbn[:l], (rbn_tensor.squeeze().detach().numpy())[:l],output[:l]
    plt.subplot(3, 1, 1)
    plt.plot(t_arr, sbn)
    plt.title('sbn')
    plt.subplot(3, 1, 2)
    plt.plot(t_arr, rbn)
    plt.title('rbn')
    plt.subplot(3, 1, 3)
    plt.plot(t_arr, output)
    plt.title('output')
    plt.show()

def main3():
    '''
    绘制使用预训练网络降噪后的信号的图像
    '''
    #声明一个模型
    model=AE1d_v3(ch=4,ker=15).double()
    #加载模型参数
    print('loading model parameters...')
    path=os.path.abspath('../model_param/ae_v12.pt')
    model.load_state_dict(torch.load(path,weights_only=True))
    model.eval()
    print('model parameters loaded')
    pre_dataset=read_dataset(mode=0,dataset_version='v4')
    pre_dataset.crt_tensor()
    t_arr,sbn,rbn_tensor=pre_dataset.t_arr,pre_dataset.sbn_tensor[-1].detach().numpy(),\
                         pre_dataset.rbn_tensor[-1].unsqueeze(0)
    l=int(len(t_arr))
    output=model(rbn_tensor).squeeze(0).detach().numpy()
    rbn=rbn_tensor[-1].detach().numpy()
    plt.subplot(3,2,1)
    plt.plot(t_arr,sbn[0])
    plt.title('sbn I branch')
    plt.subplot(3,2,2)
    plt.plot(t_arr,sbn[1])
    plt.title('sbn Q branch')
    plt.subplot(3,2,3)
    plt.plot(t_arr,rbn[0])
    plt.title('rbn I branch')
    plt.subplot(3,2,4)
    plt.plot(t_arr,rbn[1])
    plt.title('rbn Q branch')
    plt.subplot(3,2,5)
    plt.plot(t_arr,output[0])
    plt.title('output I branch')
    plt.subplot(3,2,6)
    plt.plot(t_arr,output[1])
    plt.title('output Q branch')
    plt.show()

if __name__=='__main__':
    main3()