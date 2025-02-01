# 供电图纸OCR识别

本科毕业设计，对供电工程领域的相关图纸进行OCR识别，提取关键信息用于判断相关设备是否合规以及后续操作。

## 快速开始

本项目主要使用了PaddleOCR以及QWen-VL系列模型。  
项目主要分为三个部分：
- **供电图纸OCR识别（PaddleOCR）**
- **对比实验（PaddleOCR、GOT-OCR、QWen-VL-OCR）**
- **大模型对话（QWen-VL）**

**供电图纸OCR识别**可满足基本需求，**对比实验**与**大模型对话**仅用于学习对比和模型体验。

### 模型权重

PaddleOCR轻量模型无法满足项目需求，需要使用高精度模型，可前往PaddleOCR官网下载权重，下面是我整理的模型权重。

- [百度网盘](https://pan.baidu.com/s/1BZWIDFB4O3AsOcLMAAqOfw?pwd=w8mg) 提取码: w8mg

权重文件解压后放在```weights```目录。

### 安装PaddlePaddle
根据系统配置安装指定PaddlePaddle版本。

- [PaddlePaddle](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/linux-pip.html)

### 环境配置
安装相关依赖
```bash
pip install -r requirements.txt
```
如需使用**对比实验**和**大模型对话**功能，请自行参考代码配置api-key。
### 开始运行
运行app.py文件
```bash
python app.py
```
