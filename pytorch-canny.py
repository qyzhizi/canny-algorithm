import torch
import numpy as np
from torch import nn
from PIL import Image
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
import math
import time
import argparse
from collections import deque


def dnorm(x: int, mu:int , sd:int)-> np.float32:
    return 1 / (np.sqrt(2 * np.pi) * sd) * np.e ** (-np.power((x - mu) / sd, 2) / 2)

def gaussian_kernel(size: int, sigma: int=1, verbose: bool=False) -> np.ndarray:
    kernel_1D = np.linspace(-(size // 2), size // 2, size)
    for i in range(size):
        kernel_1D[i] = dnorm(kernel_1D[i], 0, sigma)
    if verbose:
        print("kernel_1D: ", kernel_1D)
    kernel_2D = np.outer(kernel_1D, kernel_1D)
    if verbose:
        print("kernel_2D: ", kernel_2D)
    kernel_2D *= 1.0 / (kernel_2D.max() * size * size)# 归一化, 使得kernel_2D的最大值为1
    if verbose:
        print(kernel_2D.max())
        print("kernel_2D: ", kernel_2D)

    if verbose:
        plt.imshow(kernel_2D, interpolation='none', cmap='gray')
        plt.title("Kernel ( {}X{} )".format(size, size))
        plt.show()    

    return kernel_2D


# 对图像进行高斯滤波，平滑图像，去除噪声
def gaussian_blur(image: torch.tensor, kernel_size: int, verbose=False) -> torch.tensor:
    if kernel_size % 2 == 1 and kernel_size > 1 and kernel_size < 20:
        kernel = gaussian_kernel(kernel_size, sigma=math.sqrt(kernel_size), verbose=verbose)
        kernel = kernel.reshape((1, 1, kernel_size, kernel_size))
        weight = torch.from_numpy(kernel).to(torch.float32)
        # image padding kernel_size // 2, 使用原始图像的边缘像素进行填充
        image = F.pad(image, (kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2), mode='reflect')
        blur = F.conv2d(image, weight, padding=0)
        if verbose:
            plt.imshow(blur.squeeze().numpy(), cmap='gray')
            plt.title("blur")
            plt.show()
        return blur
    else:
        return image


def functional_conv2d_horizontal(im, verbose=False):
    """使用F.Conv2d进行边缘检测, 检测竖直方向的轮廓, 水平梯度，右方向为正方向

    Args:
        im (tensor): 输入的tensor图像

    Returns:
        tensor: 输出的tensor图像
    """    
    sobel_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype='float32')
    sobel_kernel = sobel_kernel.reshape((1, 1, 3, 3))
    weight = torch.from_numpy(sobel_kernel)
    im = F.pad(im, (1, 1, 1, 1), mode='replicate')
    edge_detect = F.conv2d(im, weight, stride=1, padding=0)
    if verbose:
        plt.imshow(edge_detect.squeeze().detach().numpy(), cmap='gray')
        plt.title("horizontal")
        plt.show()
    return edge_detect

def functional_conv2d_vertical(im, verbose=False):
    """使用F.Conv2d进行边缘检测, 检测水平方向的轮廓， 垂直梯度，向上为正方向

    Args:
        im (tensor): 输入的tensor图像

    Returns:
        tensor: 输出的tensor图像
    """    
    sobel_kernel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype='float32')
    sobel_kernel = sobel_kernel.reshape((1, 1, 3, 3))
    weight = torch.from_numpy(sobel_kernel)
    im = F.pad(im, (1, 1, 1, 1), mode='replicate')
    edge_detect = F.conv2d(im, weight, stride=1, padding=0)
    if verbose:
        plt.imshow(edge_detect.squeeze().detach().numpy() , cmap='gray')
        plt.title("vertical")
        plt.show()
    return edge_detect


def nms(gradient_magnitude: torch.tensor, 
        gradient_direction_quantized: torch.tensor,
        verbose=False) -> torch.tensor:
    """非极大值抑制"""

    nms_edge = torch.zeros_like(gradient_magnitude)
    nms_conv_op = torch.nn.Conv2d(in_channels=1,out_channels=3,
                                  kernel_size=3, stride=1,
                                  padding=1, bias=False)

    quantized_0_list = [[[0, 0, 0],[1, 0, 0],[0, 0, 0]],
                        [[0, 0, 0],[0, 1, 0],[0, 0, 0]],
                        [[0, 0, 0],[0, 0, 1],[0, 0, 0]]]
    
    quantized_1_list = [[[0, 1, 0],[0, 0, 0],[0, 0, 0]],
                        [[0, 0, 0],[0, 1, 0],[0, 0, 0]],
                        [[0, 0, 0],[0, 0, 0],[0, 1, 0]]]

    quantized_2_list = [[[0, 1, 0],[0, 0, 0],[0, 0, 0]],
                        [[0, 0, 0],[0, 1, 0],[0, 0, 0]],
                        [[0, 0, 0],[0, 0, 0],[0, 1, 0]]]
    
    quantized_3_list = [[[1, 0, 0],[0, 0, 0],[0, 0, 0]],
                        [[0, 0, 0],[0, 1, 0],[0, 0, 0]],
                        [[0, 0, 0],[0, 0, 0],[0, 0, 1]]]    
    
    kernel_0 = torch.tensor(quantized_0_list, dtype=torch.float32)
    kernel_1 = torch.tensor(quantized_1_list, dtype=torch.float32)
    kernel_2 = torch.tensor(quantized_2_list, dtype=torch.float32)
    kernel_3 = torch.tensor(quantized_3_list, dtype=torch.float32)
    
    kernel_direction_quantized = [kernel_0,
                                  kernel_1,
                                  kernel_2,
                                  kernel_3]
    for i in range(4):
        
        mask_i = (gradient_direction_quantized == i).int()
        if verbose:
            print("mask_i: ", mask_i)
        nms_conv_op.weight.data = kernel_direction_quantized[i].reshape((3, 1, 3, 3))
        nms_direction_conv_i = nms_conv_op(gradient_magnitude)
        nms_direction_conv_mask_i = nms_direction_conv_i * mask_i

        nms_direction_conv_mask_max_i, _ = torch.max(nms_direction_conv_mask_i, dim=1)
        if verbose:
            print("nms_direction_conv_mask_max_i: ", nms_direction_conv_mask_max_i)
            print("nms_direction_i: ", nms_direction_conv_mask_max_i)
        nms_edge = nms_edge + ((gradient_magnitude == nms_direction_conv_mask_max_i) *
                               nms_direction_conv_mask_max_i) 

    if verbose:
        print("nms_edge: ", nms_edge)

    # verbose = True
    if verbose:
        plt.imshow(nms_edge.squeeze().detach().numpy(), cmap='gray')
        plt.title("Non-Maximum Suppression")
        plt.show()
    
    return nms_edge


def get_gradient_magnitude(gradient_v: torch.tensor, gradient_h: torch.tensor, verbose=False, use_l2_norm=False) -> torch.tensor:
    """计算梯度幅值
    """
    # 计算梯度的幅值
    if use_l2_norm:
        # use l2 norm
        gradient_magnitude = torch.sqrt(gradient_v**2 + gradient_h**2)
    else:
        # use l1 norm
        gradient_magnitude = torch.abs(gradient_v) + torch.abs(gradient_h)
    # 将梯度幅值限制在0-255之间
    # gradient_magnitude = gradient_magnitude * 255.0 / gradient_magnitude.max()

    if verbose:
        plt.imshow(gradient_magnitude.squeeze().numpy(), cmap='gray')
        plt.title("Gradient Magnitude")
        plt.show()
    return gradient_magnitude


def get_gradient_direction_quantized(gradient_v: torch.tensor, gradient_h: torch.tensor, verbose=False) -> torch.tensor:
    """计算梯度方向
    将梯度方向转换为四个方向之一: 0度，45度，90度，135度, 分别用 0,1,2,3 表示。
    """

    # 计算梯度方向, 以弧度为单位
    gradient_direction = torch.atan2(gradient_v, gradient_h)

    # 将梯度方向转换为角度值, 并将角度值转换为0-360度之间
    gradient_direction_deg = torch.rad2deg(gradient_direction) + 180

    # 将角度值转换为四个方向之一：0度，45度，90度，135度
    gradient_direction_quantized = (torch.round(gradient_direction_deg / 45) % 4).int()

    if verbose:
        print("gradient_direction_quantized: ", gradient_direction_quantized)

    return gradient_direction_quantized


def threshold(image, low, high, weak=0, strong=255):
    strong_mask = image >= high
    weak_mask = (image > low) & (image < high)
    output = torch.zeros_like(image)
    output[strong_mask] = strong
    output[weak_mask] = weak
    return output


def hysteresis_threshold(nms_edge: torch.tensor, low: int, high: int) -> torch.tensor:
    nms_edge = nms_edge.squeeze()
    strong_mask = nms_edge >= high
    weak_mask = (nms_edge > low) & (nms_edge < high).int()
    output_index = []
    # get strong_mask index
    strong_index = torch.nonzero(strong_mask).tolist()
    # create stack and push strong_index
    stack = deque(strong_index)

    while len(stack) > 0:
        # pop item
        item = stack.pop()
        # get item index
        i, j = item[0], item[1]
        # get weak_mask index
        local_index = torch.nonzero(weak_mask[i-1:i+2, j-1:j+2]).tolist()
        for local_index_item in local_index:
            # stack.append(weak_index[i])
            weak_index_item = [i-1+local_index_item[0], j-1+local_index_item[1]]
            stack.append(weak_index_item)
            weak_mask[weak_index_item[0], weak_index_item[1]] = 0
        output_index.append(item)

    # create output tensor use output_index
    output_index_tensor = torch.tensor(output_index)  # (N, 2) 
    output = torch.zeros_like(nms_edge)
    output[output_index_tensor[:, 0], output_index_tensor[:, 1]] = 255

    return output


def save_edge_detect_as_image(edge_detect, save_image_path, verbose=False):
    """保存到本地
    """
    # 将tensor转换为numpy array
    if type(edge_detect) == torch.Tensor:
        edge_detect = edge_detect.squeeze().detach().numpy()

    # 将array数据转换为image
    im_image = Image.fromarray(edge_detect)

    # image数据转换为灰度模式
    im_image = im_image.convert('L')

    # 保存图片
    im_image.save(save_image_path, quality=100)


def save_edge_detect_and_origin_image_as_image(edge_detect, origin_image_path, save_image_path, verbose=False):
    """将edge_detect与原图结合，并保存为图片
    """
    # 将tensor转换为numpy array
    edge_detect = edge_detect.squeeze().detach().numpy()
    
    # 灰度图与彩色图shape一致，将灰度图当成mask，对彩色图进行处理, 选取对应的元素
    color_im_array = Image.open(origin_image_path).convert('RGB') # (H, W, C)

    color_im_array = np.array(color_im_array)
    # 改变数组的形状
    color_im_array = color_im_array.transpose((2, 0, 1))
    weak = 51
    color_im_array = color_im_array * (edge_detect >= weak)
    # 将形状改回来
    color_im_array_HWC = color_im_array.transpose((1, 2, 0)) # (H, W, C)
    if verbose:
        plt.imshow(color_im_array_HWC)
        plt.title("color_im_array")
        plt.show()

    # 将array数据转换为image
    color_im_image = Image.fromarray(color_im_array_HWC)

    # 保存图片
    color_im_image.save(save_image_path, quality=100)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--verbose", action='store_true', default=False, help="show processed image of each step")
    ap.add_argument("-i", "--image", required=True, default="lena.png", help="Path to the image")
    ap.add_argument("-gk", "--gaussian_kernel_size", type=int, default=3, help="gaussian kernel size")
    ap.add_argument("-L", "--threshold_low", type=int, default=100, help="low threshold")
    ap.add_argument("-H", "--threshold_high", type=int, default=200, help="high threshold")
    args = vars(ap.parse_args())
    verbose = args["verbose"]
    threshold_low = args["threshold_low"]
    threshold_high = args["threshold_high"]
    gaussian_kernel_size = args["gaussian_kernel_size"]
    print("args: " , args)

    # 把工作目录切换到当前文件夹所在目录
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # 打印当前工作目录
    if verbose:
        print(os.getcwd()) 

    # 读入一张图片，并转换为灰度图
    im = Image.open(args["image"]).convert('L')

    # 将图片数据转换为矩阵
    im = np.array(im, dtype='float32')

    # 将图片矩阵转换为pytorch tensor,并适配卷积输入的要求
    im = torch.from_numpy(im.reshape((1, 1, im.shape[0], im.shape[1])))
    if verbose:
        print("im.shape: ", im.shape)

    # 对图像进行高斯滤波，平滑图像，去除噪声
    im = gaussian_blur(im, gaussian_kernel_size, verbose=verbose)

    # sobel 算子进行边缘检测
    gradient_v = functional_conv2d_vertical(im, verbose=verbose) # 竖直梯度
    gradient_h = functional_conv2d_horizontal(im, verbose=verbose) # 水平梯度

    # 计算梯度幅值
    gradient_magnitude = get_gradient_magnitude(gradient_v, gradient_h, verbose=verbose)
    # 计算梯度方向
    gradient_direction_quantized = get_gradient_direction_quantized(gradient_v, gradient_h, verbose=verbose)
    # 非极大值抑制
    nms_edge = nms(gradient_magnitude, gradient_direction_quantized, verbose=verbose)

    # 普通双边阈值处理
    edge_detect = threshold(nms_edge, low=threshold_low, high=threshold_high, weak=50, strong=255)
    # 滞后阈值
    # edge_detect = hysteresis_threshold(nms_edge, low=threshold_low, high=threshold_high)

    if verbose:
        print("edge_detect.shape: ", edge_detect.shape, "im.shape: ", im.shape)
        plt.imshow(edge_detect.squeeze().detach().numpy(), cmap='gray')
        plt.title("Edge Detection")
        plt.show()

    # 保存图片
    save_edge_detect_as_image(edge_detect, "data/pytorch_edge_detect.png", verbose=verbose)
    save_edge_detect_and_origin_image_as_image(edge_detect, args["image"], 
                                               "data/pytorch_edge_detect_and_origin_image.png", verbose=verbose)
if __name__ == "__main__":
    start_time = time.time()
    print("programming strat")
    main()
    end_time = time.time()
    print("time: ", end_time - start_time)
    print("programming end")