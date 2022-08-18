# coding: UTF-8
import numpy as np
import cv2
import torch
import time

    
def CE_inside(img1 = None,img2 = None,fused = None):
    cross_entropyVI = cross_entropy(img1,fused)
    cross_entropyIR = cross_entropy(img2,fused)
    output = (cross_entropyVI + cross_entropyIR) / 2.0
    return output
    
    
def cross_entropy(img1 = None,fused = None): 
    # s = img1.shape.shape
    # if s(2) == 3:
    #     f1 = rgb2gray(img1)
    # else:
    #     f1 = img1
    f1 = img1
    
    # s1 = fused.shape.shape
    # if s1(2) == 3:
    #     f2 = rgb2gray(fused)
    # else:
    #     f2 = fused
    f2 = fused

    G1 = f1
    G2 = f2
    # G1 = f1.float()
    # G2 = f2.float()
    m1,n1 = G1.shape
    result = 0
    H1 = torch.histc(G1, bins=256, min=0, max=255)
    H2 = torch.histc(G2, bins=256, min=0, max=255)
    X1 = H1 * 1.0 / (m1 * n1)
    X2 = H2 * 1.0 / (m1 * n1)
    tmp = torch.zeros(256).to(fused.device)
    tmp[X1==0] = 100
    tmp[X2==0] = 100
    tmp = X1 * torch.log2(X1 / X2)
    tmp[X1 == 0] = 0
    tmp[X2 == 0] = 0


    result = tmp.sum()

    # for k in range(256):
    #     X1[k] = X1[k] * 1.0 / (m1 * n1)
    #     X2[k] = X2[k] * 1.0 / (m1 * n1)
    #     if (X1[k] != 0) and (X2[k] != 0):
    #         result = X1[k] * np.log2(X1[k] / X2[k]) + result
    
    res0 = result
    return res0


def CE(img1=None, img2=None, fused=None):
    # print(img1.shape)
    # (512, 512, 3)
    tmp = 0
    for i in range(3):
        tmp += CE_inside(img1[i, :, :], img2[i, :, :], fused[i, :, :])
    return tmp / 3


if __name__ == '__main__':
    img1 = torch.Tensor(cv2.imread('ue.png')).permute(2, 0, 1)
    img2 = torch.Tensor(cv2.imread('oe.png')).permute(2, 0, 1)
    fused = torch.Tensor(cv2.imread('fused.png')).permute(2, 0, 1)
    s = time.time()
    print(CE(img1, img2, fused))
    d = time.time()
    print(d-s)

