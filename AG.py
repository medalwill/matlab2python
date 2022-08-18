# coding: UTF-8
import numpy as np
import cv2
import torch


def AG(img1 = None,img2 = None,fused = None):
    fused = fused.float()
    r,c,b = fused.shape
    m1,n1,b1 = img1.shape
    m2,n2,b2 = img2.shape
    r,c,b = fused.shape
    dx = 1
    dy = 1
    tmp = 0
    for i in range(3):
        band = fused[i,:,:]
        # print(band)
        dzdx,dzdy = torch.gradient(band,spacing=[dx,dy])
        s = torch.sqrt((dzdx ** 2 + dzdy ** 2) / 2)
        tmp += sum(sum(s)) / ((b - 1) * (c - 1))
    return tmp/3


if __name__=='__main__':
    img1 = torch.Tensor(cv2.imread('ue.png')).permute(2, 0, 1)
    img2 = torch.Tensor(cv2.imread('oe.png')).permute(2, 0, 1)
    fused = torch.Tensor(cv2.imread('fused.png')).permute(2, 0, 1)
    print(AG(img1, img2, fused))
