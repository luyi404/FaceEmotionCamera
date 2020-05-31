# FaceEmotionCamera

 Machine Learning class FINAL PROJECT in **HFUT**

~~（为了逼格更高，多用英语了）~~

**数据集太大了传不上   得自己去下载**
https://www.kaggle.com/deadskull7/fer2013


The Requirement of the question I choosed:

**10.基于深度学习的面部表情分类**



**自行查找关于深度学习的资料，基于tensorflow等深度学习系统，构建一个面部表情分类，可以检测静态图片中的人脸面部表情，或者面部表情实时监测系统。**



**提交：工程报告+训练数据+工程源代码+打包程序**

What I mainly used:
* Python (Language)
* Pytorch (Frame)
* torchvision (Treat the picture data)
* openCV_python (Windowing and face finding for real-time detection)

Model : **VGG**

## Setup and Dependencies
1. Install **Anaconda** based on Python 3+
2. Clone this repository and create an environment

```
git clone https://github.com/lythings/FaceEmotionCamera
conda create -n EmotionCam python=3.6

# activate the environment and install all dependencies
conda activate EmotionCam

conda install pytorch torchvision cudatoolkit=10.2 -c pytorch

conda install opencv-python

conda install pandas
```

## Divide Data
The dataset is FER13, I got it from kaggle, All in one CSV file with labels(emotion), Pixels(Each picture is  48*48 ), I want to divide it as **Train, Validation, Test**

This  can be done as follows:
```
python DivideData.py
```

## TRAIN
I used a simplified version of the VGG model

**To visualize the model by print(model) :**
```
Sequential(
  (vggBlock_1): Sequential(
    (0): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU()
    (2): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (3): ReLU()
    (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (vggBlock_2): Sequential(
    (0): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU()
    (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (3): ReLU()
    (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (vggBlock_3): Sequential(
    (0): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU()
    (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (fc): Sequential(
    (0): FlattenLayer()
    (1): Linear(in_features=4608, out_features=1024, bias=True)
    (2): ReLU()
    (3): Dropout(p=0.5, inplace=False)
    (4): Linear(in_features=1024, out_features=1024, bias=True)
    (5): ReLU()
    (6): Dropout(p=0.5, inplace=False)
    (7): Linear(in_features=1024, out_features=7, bias=True)
  )
)
```

We can Train the model by typing   `python Train.py   `
It will save models in ./model/ after each epoch~~~~

## Run The Camera
Thanks to the openCV, It's easy to find my face in the image.
we just type 
```
python WithCamera.py
```
When you wanna close the window, Just type "q" by keyboard, then the window will close.
