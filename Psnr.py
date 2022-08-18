# coding: UTF-8

# import numpy as np
import cv2
import torch
import torchvision.transforms as transforms


def rgb2gray(im):
    transform = transforms.Grayscale()
    return transform(im)[0]


def mse(a=None, b=None):
    a = rgb2gray(a)
    b = rgb2gray(b)

    m, n = a.shape
    temp = torch.sqrt(sum(sum((a - b) ** 2)))
    res0 = temp / (m * n)
    return res0

    return res


def Psnr(img1=None, img2=None, fused=None):
    img1 = img1.float()
    img2 = img2.float()
    fused = fused.float()
    B = 8
    MAX = 2 ** B - 1
    MES = (mse(img1, fused) + mse(img2, fused)) / 2.0

    PSNR = 20 * torch.log10(MAX / torch.sqrt(MES))
    return PSNR


if __name__ == '__main__':
    img1 = torch.Tensor(cv2.imread('ue.png')).permute(2, 0, 1)
    img2 = torch.Tensor(cv2.imread('oe.png')).permute(2, 0, 1)
    fused = torch.Tensor(cv2.imread('fused.png')).permute(2, 0, 1)
    print(Psnr(img1, img2, fused))


