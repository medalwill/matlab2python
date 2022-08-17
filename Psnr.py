# coding: UTF-8

import numpy as np
import cv2


def mse(a=None, b=None):
    if a.shape[3 - 1] > 1:
        a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)

    if b.shape[3 - 1] > 1:
        b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)

    m, n = a.shape
    temp = np.sqrt(sum(sum((a - b) ** 2)))
    res0 = temp / (m * n)
    return res0

    return res


def Psnr(img1 = None,img2 = None,fused = None):
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    fused = fused.astype(np.float32)
    B = 8
    MAX = 2 ** B - 1
    MES = (mse(img1,fused) + mse(img2,fused)) / 2.0

    PSNR = 20 * np.log10(MAX / np.sqrt(MES))
    return PSNR


if __name__=='__main__':
    img1 = cv2.imread('ue.png')
    img2 = cv2.imread('oe.png')
    fused = cv2.imread('fused.png')
    print(Psnr(img1, img2, fused))
    

