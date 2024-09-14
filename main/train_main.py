from train.train import *

def main1():
    train_obj=Train(rff_dataset_version='v1',ae_version='v12',encode_channels=4,ae_kernel_size=15,
                    batch_size=8,lr=1e-5,epoch_num=100,plot_ae=True,train=False)

if __name__=='__main__':
    main1()