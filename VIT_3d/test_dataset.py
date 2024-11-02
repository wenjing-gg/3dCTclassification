import os
import nrrd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

class MyNRRDDataSet(Dataset):
    """自定义NRRD格式数据集，针对3D卷积适应 (D, H, W) 格式，并进行线性插值和分块"""

    def __init__(self, root_dir: str, split: str, transform=None, target_shape=(512, 512, 512), block_size=(256, 256, 256)):
        """
        Args:
            root_dir (str): 数据集的根目录
            split (str): 数据集划分，'train' 或 'test'
            transform (callable, optional): Optional transform to be applied on a sample.
            target_shape (tuple, optional): 目标的形状 (D, H, W)，默认填充到 (512, 512, 512).
            block_size (tuple, optional): 分块大小 (D, H, W)，默认分块为 (256, 256, 256).
        """
        self.images_path = []
        self.images_class = []
        self.transform = transform
        self.target_shape = target_shape
        self.block_size = block_size

        # 加载无转移组 (NoMetastasis) 和 有转移组 (Metastasis) 的数据
        self._load_images_from_folder(os.path.join(root_dir, split, 'NoMetastasis'), label=0)
        self._load_images_from_folder(os.path.join(root_dir, split, 'Metastasis'), label=1)

    def _load_images_from_folder(self, folder: str, label: int):
        """加载指定文件夹中的所有 NRRD 文件，并分配类别标签"""
        for filename in os.listdir(folder):
            if filename.endswith(".nrrd"):  # 假设所有文件都为图像文件
                self.images_path.append(os.path.join(folder, filename))
                self.images_class.append(label)

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        # 读取 NRRD 文件
        data, header = nrrd.read(self.images_path[item])

        # 转换为 PyTorch 张量，并确保数据格式为 (D, H, W)
        img = torch.tensor(data, dtype=torch.float32).permute(2, 0, 1)  # 将 (H, W, D) 转换为 (D, H, W)

        # 确保输入是 3D 数据 (D, H, W)
        if img.ndim != 3:
            raise ValueError(f"Image at {self.images_path[item]} is not a 3D volume.")
        
        # 线性插值到目标形状 (512, 512, 512)
        img = self.interpolate_to_shape(img, self.target_shape)

        # 分块，生成多个 (256, 256, 256) 的块
        blocks = self.split_into_blocks(img, self.block_size)

        label = self.images_class[item]

        # 如果有 transform，应用 transform 到每个块
        if self.transform:
            blocks = [self.transform(block) for block in blocks]

        # 将块的维度转换为 (C, D, H, W)，其中 C=1 表示单通道
        blocks = [block.unsqueeze(0) for block in blocks]  # 添加通道维度 (C)

        return blocks, label

    def interpolate_to_shape(self, img, target_shape):
        """
        对输入的 3D 图像进行线性插值，调整到指定形状
        Args:
            img: 输入图像张量 (D, H, W)
            target_shape: 目标形状 (target_D, target_H, target_W)
        Returns:
            调整后的张量
        """
        img = img.unsqueeze(0).unsqueeze(0)  # 添加批次和通道维度
        img = F.interpolate(img, size=target_shape, mode='trilinear', align_corners=True)  # 进行线性插值
        img = img.squeeze(0).squeeze(0)  # 移除批次和通道维度
        return img

    def split_into_blocks(self, img, block_size):
        """
        将 3D 图像按块分割
        Args:
            img: 输入图像张量 (D, H, W)
            block_size: 分块的大小 (block_D, block_H, block_W)
        Returns:
            列表，其中每个元素是一个分割块
        """
        blocks = []
        d, h, w = img.shape
        bd, bh, bw = block_size
        for i in range(0, d, bd):
            for j in range(0, h, bh):
                for k in range(0, w, bw):
                    block = img[i:i+bd, j:j+bh, k:k+bw]
                    # 如果块的大小不等于目标大小，则补零填充
                    if block.shape != block_size:
                        block = self.interpolate_to_shape(block, block_size)
                    blocks.append(block)
        return blocks

    @staticmethod
    def collate_fn(batch):
        all_blocks = []
        all_labels = []
        for blocks, label in batch:
            all_blocks.extend(blocks)  # 将所有分块加入列表
            all_labels.extend([label] * len(blocks))  # 每个分块对应的标签相同
        all_blocks = torch.stack(all_blocks, dim=0)  # 堆叠所有分块图像
        all_labels = torch.as_tensor(all_labels)  # 转换标签为张量
        return all_blocks, all_labels


def save_blocks_to_nrrd(blocks, output_dir, base_filename):
    """
    将切块后的 3D 图像块保存为 NRRD 文件
    Args:
        blocks (list of tensors): 3D 图像块列表，每个元素形状为 (C, D, H, W)，其中 C=1。
        output_dir (str): 保存块的输出目录。
        base_filename (str): 基础文件名，每个块将根据其序号命名。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 去掉通道维度 (C=1)，并保存每个块
    for idx, block in enumerate(blocks):
        block = block.squeeze(0)  # 去掉通道维度，块的形状变为 (D, H, W)
        block_np = block.cpu().numpy()  # 将张量转换为 NumPy 数组以供 NRRD 库使用
        
        # 生成保存文件名
        block_filename = os.path.join(output_dir, f"{base_filename}_block_{idx}.nrrd")
        
        # 保存 NRRD 文件
        nrrd.write(block_filename, block_np)
        print(f"Saved block {idx} to {block_filename}")


if __name__ == "__main__":
    # 使用示例
    root_dir = r'E:/肾母细胞瘤CT数据_划分'
    output_dir = r'E:/split_blocks'  # 设置保存块的输出目录

    # 创建训练集数据集，假设目标形状为 (512, 512, 512)，分块大小为 (256, 256, 256)
    train_dataset = MyNRRDDataSet(root_dir, split='train')

    # 读取第一个样本
    first_sample_blocks, first_sample_label = train_dataset[0]

    # 输出第一个样本的形状和标签
    print(f"样本块的数量: {len(first_sample_blocks)}")  # 应该有多个分块
    print(f"每个块的形状: {first_sample_blocks[0].shape}")  # 每个分块应该是 (1, 256, 256, 256)
    print(f"样本的标签: {first_sample_label}")
    print(f"Block data: {first_sample_blocks[0]}")
    
    # 打印每个块的标签
    print(f"所有块的标签: {[first_sample_label for _ in range(len(first_sample_blocks))]}")

    # 保存每个切块到 NRRD 文件
    save_blocks_to_nrrd(first_sample_blocks, output_dir, base_filename="sample_0")
