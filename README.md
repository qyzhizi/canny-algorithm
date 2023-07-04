# canny-algorithm

<div style="text-align: center;">
    <img src="https://qyzhizi.cn/img/202307041413380.png"  style="width: 50%;">
</div>

使用PyTorch 实现Canny算法，相比于纯python实现，速度会更快，因为PyTorch api 使用很多c语言的底层模块，比如卷积。另外使用Pytorch可以很方便得利用cuda的加速，所以也在GPU并行运算上也有优势。

不过相比OpenCV c++的高效，PyTorch的实现显得更冗余，效率一般。但是PyTorch 使用张量、卷积来实现图像处理，保证一定效率的同时代码更简洁，另外比较重要的是，全过程使用python语言所以程序比较方进行调试，展示算法的中间处理过程。

## 程序运行

运行命令：
```shell
python pytorch-canny.py -v -i data/lena.png -gk 0 -L 100 -H 200
```
命令行参数：
`-v`:  展示中间过程处理的图片
`-i`: 图片路径
`-gk`: 高斯核大小，默认是3， 传入0则表示不使用高斯模糊处理
`-L`: 低阈值
`-H`: 高阈值

结果：
<div>
    <img src="https://qyzhizi.cn/img/202306291745926.png" style="display: inline-block;width: 30%;">
    <img src="https://qyzhizi.cn/img/202307041413380.png"  style="display: inline-block;width: 30%;">
</div>

## more

https://qyzhizi.github.io/pytorch-implement-opencv-canny-algorithm.html
