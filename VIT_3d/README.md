## 代码使用简介
本代码旨在通过预训练的vit来对CT图像进行肾母细胞瘤的转移预测，主要将第一层embedding改为三维卷积。
1. 下载好数据集，代码中默认使用的是转移分类数据集，下载地址: https://pan.baidu.com/s/1Z1vIXOuhgl8qLRioYPZkKg?pwd=b0pw
   文件路径格式应设置为(其中以label结尾的文件为肿瘤mask，在本代码下尚未使用，需删除)
   /home/yuwenjing/data/肾母细胞瘤CT数据_划分/
    ├── train/
    │   ├── NoMetastasis/
    │   │   ├── 1_image.nrrd
    │   │   ├── 1_label.nrrd
    │   │   └── ...
    │   ├── Metastasis/
    │   │   ├── 2_image.nrrd
    │   │   ├── 2_label.nrrd
    │   │   └── ...
    └── test/
        ├── NoMetastasis/
        │   ├── 3_image.nrrd
        │   ├── 3_label.nrrd
        │   └── ...
        └── Metastasis/
            ├── 4_image.nrrd
            ├── 4_label.nrrd
            └── ...

3. 在`train.py`脚本中将`--data-path`设置成解压后的`flower_photos`文件夹绝对路径
4. 下载预训练权重，在`vit_model.py`文件中每个模型都有提供预训练权重的下载地址，根据自己使用的模型下载对应预训练权重，仅使用了前几层
5. 在`train.py`脚本中将`--weights`参数设成下载好的预训练权重路径
6. 设置好数据集的路径`--data-path`以及预训练权重的路径`--weights`就能使用`train.py`脚本开始训练了(训练过程中会自动生成`class_indices.json`文件)
7. `predict.py`尚不可用


