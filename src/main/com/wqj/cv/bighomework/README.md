# 2021CVORDL
期末大作业

1.先安装 py3.6.5 ,显卡驱动,cuda


2.安装依赖环境
   pip3 install tensorflow
   pip3 install tensorflow-gpu

   pip3 install   keras
   pip3 install  matplotlib

   pip3 install  pillow
    
    
 


3.1 对database文件夹内图片进行特征提取，建立索引文件featureCNN.h5
    python index.py -database database -index featureCNN.h5

3.2 使用database文件夹内001_accordion_image_0001.jpg作为测试图片，在database内以featureCNN.h5进行近似图片查找，并显示最近似的3张图片
    python query_online.py -query database/001_accordion_image_0001.jpg -index featureCNN.h5 -result database

