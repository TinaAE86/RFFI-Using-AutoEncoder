from dataset_gnt.dataset_gnt import DatasetGnt
import math

pi=math.pi

def main1():
    dataset_gnt = DatasetGnt(T=1.024e-3, N=8192, fc=868e6, SF=7, M=20,
                             beta1=1, beta2=1.1, phi1=-pi / 40, phi2=pi / 40, c21=-0.1, c22=0, c31=-0.1, c32=0.1)
    n, gamma_s, gamma_r = 1000, 70, 20
    dataset_gnt.gnt_pre_dataset_v3(n=n, gamma_s=gamma_s, gamma_r=gamma_r, pre_dataset_version='v4')
    dataset_gnt.gnt_rff_dataset(n=n, gamma_r=gamma_r, rff_dataset_version='v1')

if __name__=='__main__':
    main1()