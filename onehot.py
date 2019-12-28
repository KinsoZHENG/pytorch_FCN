import numpy as np
import cv2


def onehot(data, n):
    buf = np.zeros(data.shape + (n,))
    nmsk = np.arange(data.size) * n + data.ravel()
    buf.ravel()[nmsk - 1] = 1
    # print(buf.shape)
    return buf


if __name__ == '__main__':
    img = cv2.imreadmulti(
        "/data_msk/EAD2020_semantic_00001.tif",
        None, -1)
    # cv2.im
    print(len(img[1]))
    buf = []
    for idx in range(len(img[1])):
        print(idx)
        imgB = cv2.resize(img[1][idx], (160, 160))
        print(imgB.shape)
        imgB = imgB / 255
        imgB = imgB.astype('uint8')
        imgB = onehot(imgB, 5)
        # imgB = imgB.transpose(2, 0, 1)
        # imgB = torch.FloatTensor(imgB)
        buf.append(imgB)

    print(buf)
