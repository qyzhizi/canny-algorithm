# opencv canny
import cv2
import numpy as np
from matplotlib import pyplot as plt
import time

if __name__ == '__main__':
    img = cv2.imread('lena.png',0)
    start_time = time.time()
    edges = cv2.Canny(img,80,100)
    print("time: ", time.time() - start_time)
    plt.subplot(121),plt.imshow(img,cmap = 'gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(edges,cmap = 'gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    plt.show()
    print("end")

    # 保存图片
    cv2.imwrite("opencv-lena-result.png", edges)
    

