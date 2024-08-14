import os
import timeit
from datetime import datetime
import socket
import time
from torch.autograd import Variable

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import VideoDataset

import C3D_model
from tensorboardX import SummaryWriter

def train_model(num_epoches, num_classes, lr, device, save_dir, train_dataloader, val_dataloader, test_dataloader):
    model = C3D_model.C3D(num_classes, prtrained=True)
    # 损失定义
    criterion = nn.CrossEntropyLoss()
    # 优化器
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    #  定义学习率更新策略    不断更新学习率大小，从而使得不至于跳的那么大。 10轮更新一次，大小为*0.1
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    # 放置模型，损失计算放在设备之中
    model.to(device)
    criterion.to(device)
    # 日记路径
    log_dir = os.path.join(save_dir, 'models', datetime.now().strftime('%b%d_%H-%M-%S') + '-' + socket.gethostname())
    writer = SummaryWriter(log_dir=log_dir)

    # 开始训练
    trainval_loaders = {'train': train_dataloader, 'val': val_dataloader}
    trainval_size = {x:len(trainval_loaders[x].dataset) for x in ['train', 'val']}

    test_size = len(test_dataloader.dataset)

    for epoch in range(num_epoches):
        for phase in ['train', 'val']:
            start_time = timeit.default_timer() # 记录开始时间

            running_loss = 0.0
            running_corrects = 0.0
            # 初始loss和精度

            if phase == 'train':
                model.train()
            else:
                model.eval()
            for inputs, labels in tqdm(trainval_loaders[phase]): # 使得可视化。
                inputs = Variable(inputs, requires_grad=True).to(device)
                # 这种其实已经out了
                labels = Variable(labels).to(device)
                optimizer.zero_grad()

                if phase == 'train':
                    outputs = model(inputs)
                else:
                    with torch.no_grad():
                        outputs = model(inputs)
                # 如果是训练就开始进行梯度累加，不是的话就不进行，这样子可以保证内存不会被占用
                probs = nn.Softmax(dim=1)(outputs)
                # 得到结果概率最大值
                preds = torch.max(probs, 1)[1]
                labels = labels.long()
                # 必须是long，否则会报错
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item() * inputs.size(0)

                running_corrects += torch.sum(preds == labels.data)
            scheduler.step()  # 训练结束开始修改学习率
            epoch_loss = running_loss / trainval_size[phase]  # 计算该轮次的损失值
            epoch_acc = running_corrects.double() / trainval_size[phase]  # 计算该轮次的准确值
            if phase == 'train':
                writer.add_scalar('data/train-loss-epoch', epoch_loss, epoch)
                writer.add_scalar('data/train-acc-epoch', epoch_acc, epoch)
            else:
                writer.add_scalar('data/val-loss-epoch', epoch_loss, epoch)
                writer.add_scalar('data/val-acc-epoch', epoch_acc, epoch)

            stop_time = timeit.default_timer()
            print("[{}] Epoch: {}/{} Loss: {} Acc:{}".format(phase, epoch + 1, num_epoches, epoch_loss, epoch_acc))
            print("Execution time: " + str(stop_time - start_time) + '\n')

    writer.close()
    # 保存模型
    torch.save({'epoch': epoch + 1, 'state-dict': model.state_dict(), 'opt_dict': optimizer.state_dict(), },
               os.path.join(save_dir, 'models', 'C3D' + '_epoch-' + str(epoch) + '.pth.tar'))
    print("Save model at {}\n".format(os.path.join(save_dir, 'models', 'C3D' + '_epoch-' + str(epoch) + '.pth.tar')))

    # 开始模型测试
    model.eval()
    running_corrects = 0.0
    for inputs, labels in tqdm(test_dataloader):  # 使得可视化。
        inputs = Variable(inputs, requires_grad=True).to(device)
        # 这种其实已经out了
        labels = labels.long()
        labels = Variable(labels).to(device)
        optimizer.zero_grad()
        with torch.no_grad():
            output = model(inputs)
        probs = nn.Softmax(dim=1)(outputs)
        preds = torch.max(probs, 1)[1]
        running_corrects += torch.sum(preds == labels.data)
    epoch_acc = running_corrects.double() / test_size
    print("Test Acc: {}".format(epoch_acc))

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_epochs = 20
    lr = 1e-3
    num_classes = 101
    save_dir = 'model_result'

    train_dataloader = DataLoader(VideoDataset(dataset_path='data/ucf101', images_path='train', clip_len=16), batch_size=16, shuffle=True, num_workers=2)
    val_dataloader = DataLoader(VideoDataset(dataset_path='data/ucf101', images_path='val', clip_len=16), batch_size=16, shuffle=True, num_workers=2)
    test_dataloader = DataLoader(VideoDataset(dataset_path='data/ucf101', images_path='test', clip_len=16), batch_size=16, shuffle=True, num_workers=2)

    train_model(num_epochs, num_classes, lr, device, save_dir, train_dataloader, val_dataloader, test_dataloader)