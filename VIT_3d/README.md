# 代码使用简介

本代码旨在通过预训练的 Vision Transformer (ViT) 模型对 CT 图像进行肾母细胞瘤的转移预测，主要对 ViT 的第一层 embedding 进行修改，使其支持三维卷积。

## 使用步骤

### 1. 下载数据集

代码默认使用的是肾母细胞瘤转移分类数据集。下载地址：[百度网盘](https://pan.baidu.com/s/1Z1vIXOuhgl8qLRioYPZkKg?pwd=b0pw)

下载并解压后，将数据集设置为以下目录结构：

/home/yuwenjing/data/肾母细胞瘤CT数据_划分/ ├── train/ │ ├── NoMetastasis/ │ │ ├── 1_image.nrrd │ │ ├── 1_label.nrrd │ │ └── ... │ ├── Metastasis/ │ │ ├── 2_image.nrrd │ │ ├── 2_label.nrrd │ │ └── ... └── test/ ├── NoMetastasis/ │ ├── 3_image.nrrd │ ├── 3_label.nrrd │ └── ... └── Metastasis/ ├── 4_image.nrrd ├── 4_label.nrrd └── ...


> **注意**：在上述路径中，以 `_label` 结尾的文件为肿瘤 mask。代码当前未使用这些文件，因此需要将其删除。

### 2. 设置数据集路径

在 `train.py` 脚本中，将 `--data-path` 参数设置为解压后的数据集文件夹的绝对路径，例如 `/home/yuwenjing/data/肾母细胞瘤CT数据_划分/`。

### 3. 下载预训练权重

在 `vit_model.py` 文件中，每个模型提供了预训练权重的下载地址。根据使用的模型版本下载对应的预训练权重。代码仅使用了权重的前几层。

### 4. 设置预训练权重路径

在 `train.py` 脚本中，将 `--weights` 参数设为下载好的预训练权重的路径。

### 5. 开始训练

设置好数据集路径 `--data-path` 和预训练权重路径 `--weights` 后，可以运行 `train.py` 脚本开始训练。在训练过程中会自动生成 `class_indices.json` 文件。

### 6. 注意事项

- 当前 `predict.py` 脚本尚不可用，可以忽略预测脚本的部分。
