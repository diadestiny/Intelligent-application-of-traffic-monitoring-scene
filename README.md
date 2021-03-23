# Intelligent-application-of-traffic-monitoring-scene

一、运行所需环境、相关库，以及配置

1. 运行系统：windows

2. Python版本要求：3.7

3. 需要cuda、cudnn以及对应的torch-1.2.0版本和torchvision-0.4.0版本

比如（满足cuda、cudnn和torch-1.2.0、torchvison-0.4.0版本对应即可）：

(1)windows版本的cudn9.2以及cudnn9.2, torch-1.2.0+cu92-cp37-cp37m-win_amd64.

whl和torchvision-0.4.0+cu92-cp37-cp37m-win_amd64.whl

(2)windows版本的cudn10.0以及cudnn10.0, torch-1.2.0-cp37-cp37m-win_amd64.whl

和torchvision-0.4.0-cp37-cp37m-win_amd64.whl

4. 需要通过pip等方式安装opencv_python,albumentations,imutils,torch(要求如

 上), torchvision(要求如上)五个依赖库。

 

 

 

二、本程序使用说明书

1. 以上环境配置完成后，用PyCharm等工具打开AACarTeam项目文件夹，运行GUI.py

启动项目。

![img](file:///C:/Users/lin/AppData/Local/Temp/msohtmlclip1/01/clip_image002.jpg)

 

 

2. 点击“选择视频”按钮，选择项目下videos文件夹下的测试视频。

(这里以video-03.avi为例)

![img](file:///C:/Users/lin/AppData/Local/Temp/msohtmlclip1/01/clip_image004.jpg)

 

3. 点击“开始识别”按钮。