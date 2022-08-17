import numpy as np
import cv2


def mutual_info(im1=None, im2=None):

    im1 = im1.astype(np.float32)
    im2 = im2.astype(np.float32)
    hang, lie = im1.shape
    count = hang * lie
    N = 256
    ## caculate the joint histogram
    h = np.zeros((N, N))
    for i in range(hang):
        for j in range(lie):
            # in this case im1->x (row), im2->y (column)
            h[int(im1[i, j]), int(im2[i, j])] = h[int(im1[i, j]), int(im2[i, j])] + 1

    ## marginal histogram

    # this operation converts histogram to probability
    # h=h./count;
    h = h / np.sum(h)
    im1_marg = np.sum(h,0)

    im2_marg = np.sum(np.transpose(h),0)

    H_x = - np.sum(np.multiply(im1_marg, np.log2(im1_marg + (im1_marg == 0))))
    H_y = - np.sum(np.multiply(im2_marg, np.log2(im2_marg + (im2_marg == 0))))
    # joint entropy
    H_xy = - np.sum(np.sum(np.multiply(h, np.log2(h + (h == 0)))))
    # mutual information
    MI = H_x + H_y - H_xy
    # print(MI)
    # print(H_xy)
    # print(H_x)
    # print(H_y)
    return MI, H_xy, H_x, H_y

    return res


def normalize1(i):
    i = i.astype(np.float32)
    da = np.max(i)
    xiao = np.min(i)
    if da == 0 and xiao == 0:
        return i
    else:
        newdata = (i - xiao) / (da - xiao)
        return np.round(newdata * 255)


def NMI(im1 = None,im2 = None,fused = None): 

    ## pre-processing
    im1 = normalize1(im1)
    im2 = normalize1(im2)
    fused = normalize1(fused)

    I_fx,H_xf,H_x,H_f1 = mutual_info(im1,fused)
    I_fy,H_yf,H_y,H_f2 = mutual_info(im2,fused)
    MI = 2 * (I_fx / (H_f1 + H_x) + I_fy / (H_f2 + H_y))
    output = MI
    return output


if __name__ == '__main__':
    img1 = cv2.imread('ue.png')
    img2 = cv2.imread('oe.png')
    fused = cv2.imread('fused.png')
    # print(img1.shape)
    # (512, 512, 3)
    sum = 0
    for i in range(3):
        sum += NMI(img1[:,:,i], img2[:,:,i], fused[:,:,i])
    print(sum/3)
