# coding: UTF-8
import numpy as np
import cv2
import torch


def SD_inside(img1=None, img2=None, fused=None):
    # if fused.shape[3-1] > 1:
    #     fused = cv2.cvtColor(fused, cv2.COLOR_BGR2GRAY)

    fused = fused.float()
    r, c = fused.shape
    # Mean value
    img_mean = torch.mean(fused)
    # Variance
    img_var = torch.sqrt(sum(sum((fused - img_mean) ** 2)) / (r * c))
    return img_mean, img_var


def SD(img1=None, img2=None, fused=None):
    # print(img1.shape)
    # (512, 512, 3)
    tmp = 0
    for i in range(3):
        tmp += SD_inside(img1[i, :, :], img2[i, :, :], fused[i, :, :])[1]
    return tmp / 3


if __name__ == '__main__':
    img1 = torch.Tensor(cv2.imread('ue.png')).permute(2, 0, 1)
    img2 = torch.Tensor(cv2.imread('oe.png')).permute(2, 0, 1)
    fused = torch.Tensor(cv2.imread('fused.png')).permute(2, 0, 1)
    print(SD(img1, img2, fused))