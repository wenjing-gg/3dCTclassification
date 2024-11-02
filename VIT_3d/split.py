import os
import random
import shutil

def split_dataset(root_dir, output_dir, train_ratio=0.8):
    """
    将数据集划分为训练集和测试集，并将结果复制到指定目录结构中
    仅处理文件名中带有 'image' 的文件，忽略 'label' 文件。
    
    Args:
        root_dir (str): 原始数据集的根目录
        output_dir (str): 保存划分结果的目录
        train_ratio (float): 训练集比例，默认 0.8
    """
    # 定义类别组，Data0 和 Data3-0 归为无转移，Data1 和 Data3-1 归为有转移
    class_groups = {
        'NoMetastasis': ['Data0', 'Data3-0'],  # 无转移组
        'Metastasis': ['Data1', 'Data3-1']     # 有转移组
    }
    
    # 创建输出目录结构
    train_dir = os.path.join(output_dir, "train")
    test_dir = os.path.join(output_dir, "test")
    for c in class_groups.keys():
        os.makedirs(os.path.join(train_dir, c), exist_ok=True)
        os.makedirs(os.path.join(test_dir, c), exist_ok=True)
    
    for class_name, folders in class_groups.items():
        # 获取每个类别组中的所有样本，只选择以 'image' 结尾的文件
        samples = []
        for folder in folders:
            folder_path = os.path.join(root_dir, folder)
            # 筛选文件名中包含 'image' 的文件
            samples += [os.path.join(folder_path, f) for f in os.listdir(folder_path) 
                        if f.endswith(".nrrd") and 'image' in f]

        # 打乱顺序并按比例分割
        random.shuffle(samples)
        split_point = int(len(samples) * train_ratio)
        train_samples = samples[:split_point]
        test_samples = samples[split_point:]

        print(f"类别 {class_name}: 训练集 {len(train_samples)} 样本, 测试集 {len(test_samples)} 样本")

        # 将文件复制到训练集和测试集目录
        for sample in train_samples:
            shutil.copy(sample, os.path.join(train_dir, class_name, os.path.basename(sample)))
        
        for sample in test_samples:
            shutil.copy(sample, os.path.join(test_dir, class_name, os.path.basename(sample)))

    print("数据集划分完成并已复制到:", output_dir)

# 使用示例
if __name__ == "__main__":
    root_dir = r"E:\肾母细胞瘤CT数据"  # 原始数据集路径
    output_dir = r"E:\肾母细胞瘤CT数据_划分"  # 保存划分结果的路径
    split_dataset(root_dir, output_dir, train_ratio=0.8)  # 80% 训练集，20% 测试集
