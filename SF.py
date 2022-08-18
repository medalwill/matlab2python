# coding: UTF-8
import numpy as np
import cv2
import torch
import time
    
    
def SF_inside(img1 = None,img2 = None,fused = None):

    fused = fused.float()
    m,n = fused.shape

    tmp1 = torch.zeros([m, n]).to(fused.device)
    tmp2 = torch.zeros([m, n]).to(fused.device)
    tmp1[:, 1:] = fused[:, :-1]
    tmp1[:, 0] = fused[:, 0]
    tmp2[1:, :] = fused[:-1, :]
    tmp2[0, :] = fused[0, :]
    RF = torch.sum((fused - tmp1) ** 2) / (m * n)
    CF = torch.sum((fused - tmp2) ** 2) / (m * n)
    # RF = 0
    # CF = 0
    # for fi in range(m):
    #     for fj in range(1,n):
    #         RF = RF + (fused[fi,fj] - fused[fi,fj - 1]) ** 2
    #
    # RF = RF / (m * n)
    # for fj in range(n):
    #     for fi in range(1,m):
    #         CF = CF + (fused[fi,fj] - fused[fi - 1,fj]) ** 2
    #
    # CF = CF / (m * n)

    output = torch.sqrt(RF + CF)

    return output


def SF(img1=None, img2=None, fused=None):
    # print(img1.shape)
    # (512, 512, 3)
    tmp = 0
    for i in range(3):
        tmp += SF_inside(img1[i, :, :], img2[i, :, :], fused[i, :, :])
    return tmp / 3


if __name__ == '__main__':
    img1 = torch.Tensor(cv2.imread('ue.png')).permute(2, 0, 1)
    img2 = torch.Tensor(cv2.imread('oe.png')).permute(2, 0, 1)
    fused = torch.Tensor(cv2.imread('fused.png')).permute(2, 0, 1)
    s = time.time()
    print(SF(img1, img2, fused))
    d = time.time()
    print(d-s)

