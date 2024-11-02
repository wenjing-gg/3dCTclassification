# import os
# import math
# import argparse

# import torch
# import torch.optim as optim
# import torch.optim.lr_scheduler as lr_scheduler
# from torch.utils.tensorboard import SummaryWriter
# from torchvision import transforms

# from test_dataset import MyNRRDDataSet  # 使用您自定义的数据集类
# from vit_3d import vit_base_patch16_test as create_model  # 确保模型为3D版本
# from utils import train_one_epoch, evaluate
# from torch.cuda.amp import GradScaler



# def main(args):
#     device = torch.device(args.device if torch.cuda.is_available() else "cpu")
#     scaler = GradScaler()  # 初始化混合精度梯度缩放器
#     if not os.path.exists("./weights"):
#         os.makedirs("./weights")

#     tb_writer = SummaryWriter()

#     data_transform = {
#         "train": transforms.Compose([
#             transforms.Normalize([0.5], [0.5])  # 假设单通道数据
#         ]),
#         "val": transforms.Compose([
#             transforms.Normalize([0.5], [0.5])
#         ])
#     }

#     # 实例化训练数据集
#     train_dataset = MyNRRDDataSet(root_dir=args.data_path, split='train', transform=data_transform["train"], target_shape=(512, 512, 512), block_size=(256, 256, 256))

#     # 实例化验证数据集
#     val_dataset = MyNRRDDataSet(root_dir=args.data_path, split='test', transform=data_transform["val"], target_shape=(512, 512, 512), block_size=(256, 256, 256))

#     batch_size = args.batch_size
#     # nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
#     nw = 8
#     print(f'Using {nw} dataloader workers every process')

#     # 更新dataloader，将collate_fn传入，以便处理分块后的数据
#     train_loader = torch.utils.data.DataLoader(train_dataset,
#                                                batch_size=batch_size,  # 批次大小指每次训练的块数
#                                                shuffle=True,
#                                                pin_memory=True,
#                                                num_workers=nw,
#                                                collate_fn=train_dataset.collate_fn)

#     val_loader = torch.utils.data.DataLoader(val_dataset,
#                                              batch_size=batch_size,  # 验证批次同样指块数
#                                              shuffle=False,
#                                              pin_memory=True,
#                                              num_workers=nw,
#                                              collate_fn=val_dataset.collate_fn)

#     # 使用支持3D输入的Vision Transformer模型
#     model = create_model(num_classes=args.num_classes, has_logits=False).to(device)

#     if args.weights != "":
#         assert os.path.exists(args.weights), f"weights file: '{args.weights}' not exist."
#         weights_dict = torch.load(args.weights, map_location=device,weights_only=True)
#         # 删除不需要的权重
#         del_keys = ['head.weight', 'head.bias'] if model.has_logits else ['pre_logits.fc.weight', 'pre_logits.fc.bias', 'head.weight', 'head.bias', 'pos_embed', 'patch_embed.proj.weight', 'patch_embed.proj.bias']
#         for k in del_keys:
#             del weights_dict[k]
#         print(model.load_state_dict(weights_dict, strict=False))


#     if args.freeze_layers:
#         for name, para in model.named_parameters():
#             # 除head, pre_logits外，其他权重全部冻结
#             if "head" not in name and "pre_logits" not in name:
#                 para.requires_grad_(False)
#             else:
#                 print(f"training {name}")

#     pg = [p for p in model.parameters() if p.requires_grad]
#     # 使用 Adam 优化器代替 SGD
#     optimizer = optim.Adam(pg, lr=args.lr, betas=(0.9, 0.999), weight_decay=10E-8)

    
#     # 使用余弦退火学习率调度器
#     lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
#     scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

#     block_batch_size = args.block_batch_size
#     for epoch in range(args.epochs):
#         # train
#         train_loss, train_acc = train_one_epoch(model=model,
#                                                 optimizer=optimizer,
#                                                 data_loader=train_loader,
#                                                 device=device,
#                                                 epoch=epoch,
#                                                 scaler=scaler,
#                                                 block_batch_size=block_batch_size)

#         scheduler.step()

#         # validate
#         val_loss, val_acc = evaluate(model=model,
#                                      data_loader=val_loader,
#                                      device=device,
#                                      epoch=epoch)

#         tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
#         tb_writer.add_scalar(tags[0], train_loss, epoch)
#         tb_writer.add_scalar(tags[1], train_acc, epoch)
#         tb_writer.add_scalar(tags[2], val_loss, epoch)
#         tb_writer.add_scalar(tags[3], val_acc, epoch)
#         tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

#         torch.save(model.state_dict(), f"./weights/model-{epoch}.pth")

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--num_classes', type=int, default=2)  # 假设2个分类：无转移和有转移
#     parser.add_argument('--epochs', type=int, default=20)
#     parser.add_argument('--batch-size', type=int, default=1)  # 这里的batch-size是每次加载的块数，不是样本数
#     parser.add_argument('--block-batch-size', type=int, default=2, help="每次传入模型的块数")  # 设置 block_batch_size 超参数
#     parser.add_argument('--lr', type=float, default=0.00001)
#     parser.add_argument('--lrf', type=float, default=0.01)

#     # 数据集所在根目录
#     parser.add_argument('--data-path', type=str, default=r"/home/yuwenjing/data/肾母细胞瘤CT数据_划分")  # 更新路径

#     # 预训练权重路径，如果不想载入就设置为空字符
#     parser.add_argument('--weights', type=str, default='./vit_base_patch16_224_in21k.pth', help='initial weights path')

#     # 是否冻结权重
#     parser.add_argument('--freeze-layers', type=bool, default=False)
#     parser.add_argument('--device', default='cuda:2', help='device id (i.e. 0 or 0,1 or cpu)')

#     opt = parser.parse_args()
    
#     main(opt)

import os
import math
import argparse

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from test_dataset import MyNRRDDataSet  # 使用您自定义的数据集类
from vit_3d import vit_base_patch16_test as create_model  # 确保模型为3D版本
from utils import train_one_epoch, evaluate
from torch.cuda.amp import GradScaler


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    scaler = GradScaler()  # 初始化混合精度梯度缩放器
    if not os.path.exists("./weights"):
        os.makedirs("./weights")

    tb_writer = SummaryWriter()

    data_transform = {
        "train": transforms.Compose([
            transforms.Normalize([0.5], [0.5])  # 假设单通道数据
        ]),
        "val": transforms.Compose([
            transforms.Normalize([0.5], [0.5])
        ])
    }

    # 实例化训练数据集
    train_dataset = MyNRRDDataSet(root_dir=args.data_path, split='train', transform=data_transform["train"], target_shape=(512, 512, 512), block_size=(256, 256, 256))

    # 实例化验证数据集
    val_dataset = MyNRRDDataSet(root_dir=args.data_path, split='test', transform=data_transform["val"], target_shape=(512, 512, 512), block_size=(256, 256, 256))

    batch_size = args.batch_size
    nw = 8
    print(f'Using {nw} dataloader workers every process')

    # 更新dataloader，将collate_fn传入，以便处理分块后的数据
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,  # 批次大小指每次训练的块数
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,  # 验证批次同样指块数
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)

    # 使用支持3D输入的Vision Transformer模型
    model = create_model(num_classes=args.num_classes, has_logits=False).to(device)

    if args.weights != "":
        assert os.path.exists(args.weights), f"weights file: '{args.weights}' not exist."
        weights_dict = torch.load(args.weights, map_location=device, weights_only=True)
        # 删除不需要的权重
        del_keys = ['head.weight', 'head.bias'] if model.has_logits else ['pre_logits.fc.weight', 'pre_logits.fc.bias', 'head.weight', 'head.bias', 'pos_embed', 'patch_embed.proj.weight', 'patch_embed.proj.bias']
        for k in del_keys:
            del weights_dict[k]
        print(model.load_state_dict(weights_dict, strict=False))

    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除head, pre_logits外，其他权重全部冻结
            if "head" not in name and "pre_logits" not in name:
                para.requires_grad_(False)
            else:
                print(f"training {name}")

    pg = [p for p in model.parameters() if p.requires_grad]
    # 使用 Adam 优化器代替 SGD
    optimizer = optim.Adam(pg, lr=args.lr, betas=(0.9, 0.999), weight_decay=10E-8)

    # 使用余弦退火学习率调度器
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    block_batch_size = args.block_batch_size
    for epoch in range(args.epochs):
        # train
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch,
                                                scaler=scaler,
                                                block_batch_size=block_batch_size)

        scheduler.step()

        # validate
        val_loss, val_acc = evaluate(model=model,
                                     data_loader=val_loader,
                                     device=device,
                                     epoch=epoch)

        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

        torch.save(model.state_dict(), f"./weights/model-{epoch}.pth")

        # 打印经过 transforms 处理后的数据
        for data, target in train_loader:
            print("经过 transforms 处理后的数据:", data)
            break  # 只打印一个批次的数据

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=2)  # 假设2个分类：无转移和有转移
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=1)  # 这里的batch-size是每次加载的块数，不是样本数
    parser.add_argument('--block-batch-size', type=int, default=2, help="每次传入模型的块数")  # 设置 block_batch_size 超参数
    parser.add_argument('--lr', type=float, default=0.00001)
    parser.add_argument('--lrf', type=float, default=0.01)

    # 数据集所在根目录
    parser.add_argument('--data-path', type=str, default=r"/home/yuwenjing/data/肾母细胞瘤CT数据_划分")  # 更新路径

    # 预训练权重路径，如果不想载入就设置为空字符
    parser.add_argument('--weights', type=str, default='./vit_base_patch16_224_in21k.pth', help='initial weights path')

    # 是否冻结权重
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:2', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()
    
    main(opt)