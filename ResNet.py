import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import argparse
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from PIL import Image
import os
import torch.optim as optim
from tqdm import tqdm

# 超参数
BATCH_SIZE = 64  # 每批处理的数据 一次性多少个
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 使用GPU
# 打印当前使用的设备
if DEVICE.type == 'cuda':
    print(f"CUDA设备已启用，使用的设备是: {torch.cuda.get_device_name(DEVICE)}")
else:
    print("CUDA设备不可用，使用CPU进行训练。")

EPOCHS = 100  # 训练数据集的轮次
model_filename = f'model{EPOCHS}.pth'


# 构建网络模型
class ResidualBlock(nn.Module):  # 定义Resnet Block模块

    def __init__(self, inchannel, outchannel, stride=1):  # 进入网络前先得知道传入层数和传出层数的设定

        super(ResidualBlock, self).__init__()  # 初始化

        # 根据resnet网络结构构建2个（block）块结构 第一层卷积 卷积核大小3*3,步长为1，边缘加1
        self.conv1 = nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1)
        # 将第一层卷积处理的信息通过BatchNorm2d
        self.bn1 = nn.BatchNorm2d(outchannel)
        # 第二块卷积接收第一块的输出，操作一样
        self.conv2 = nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(outchannel)

        # 确保输入维度等于输出维度
        self.extra = nn.Sequential()  # 先建一个空的extra
        if outchannel != inchannel:
            # [b, ch_in, h, w] => [b, ch_out, h, w]
            self.extra = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):  # 定义局部向前传播函数
        out = F.relu(self.bn1(self.conv1(x)))  # 对第一块卷积后的数据再经过relu操作
        out = self.bn2(self.conv2(out))  # 第二块卷积后的数据输出
        out = self.extra(x) + out  # 将x传入extra经过2块（block）输出后与原始值进行相加
        out = F.relu(out)  # 调用relu
        return out


class ResNet18(nn.Module):  # 构建resnet18层

    def __init__(self):
        super(ResNet18, self).__init__()

        self.conv1 = nn.Sequential(  # 首先定义一个卷积层
            nn.Conv2d(1, 32, kernel_size=3, stride=3, padding=0),
            nn.BatchNorm2d(32)
        )
        # followed 4 blocks 调用4次resnet网络结构，输出都是输入的2倍
        self.layer1 = ResidualBlock(32, 64, stride=1)
        self.layer2 = ResidualBlock(64, 128, stride=1)
        self.layer3 = ResidualBlock(128, 256, stride=1)
        self.layer4 = ResidualBlock(256, 256, stride=1)
        self.outlayer = nn.Linear(256 * 1 * 1, 10)  # 最后是全连接层

    def forward(self, x):  # 定义整个向前传播

        x = F.relu(self.conv1(x))  # 先经过第一层卷积

        x = self.layer1(x)  # 然后通过4次resnet网络结构
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = F.adaptive_avg_pool2d(x, [1, 1])
        # print('after pool:', x.shape)
        x = x.view(x.size(0), -1)  # 平铺一维值
        x = self.outlayer(x)  # 全连接层

        return x


def train_model(model, device, train_loader, optimizer, criterion, epoch):
    model.train()  # 设置为训练模式
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  # 学习率调度器
    for batch_index, (data, target) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()  # 清除旧的梯度
        output = model(data)  # 前向传播
        loss = criterion(output, target)  # 计算损失
        loss.backward()  # 反向传播

        # 可选：梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 最大梯度范数为 1.0

        optimizer.step()  # 更新参数

        if batch_index % 32 == 0:  # 每 5 个 batch 打印一次进度
            print(f'Train Epoch: {epoch} [{batch_index * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_index / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

    scheduler.step()  # 更新学习率


# 测试
def test_model(model, device, text_loader, criterion):
    model.eval()  # 模型验证
    correct = 0.0  # 正确率
    global Accuracy
    text_loss = 0.0
    with torch.no_grad():
        for data, target in text_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)  # 处理后的结果
            text_loss += criterion(output, target).item()  # 计算测试损失
            pred = output.argmax(dim=1)  # 找到概率最大的下标
            correct += pred.eq(target.view_as(pred)).sum().item()  # 累计正确的值
        text_loss /= len(text_loader.dataset)  # 损失和/加载的数据集的总数
        Accuracy = 100.0 * correct / len(text_loader.dataset)
        print("Test__Average loss: {:4f},Accuracy: {:.3f}\n".format(text_loss, Accuracy))


def load_model(model, model_path):
    try:
        model.load_state_dict(torch.load(model_path))
        print("Loaded model from checkpoint")
    except FileNotFoundError:
        print("No checkpoint found, starting from scratch")


def save_model(model, filename):
    torch.save(model.state_dict(), filename)
    print(f'Model saved to {filename}')


def show1(model, test_loader, DEVICE):
    model.eval()  # 将模型设置为评估模式
    for (n, (x, y)) in enumerate(test_loader):
        if n > 10:
            break
        x = x.to(DEVICE)
        with torch.no_grad():
            # 获取模型预测
            predict = model(x).argmax(dim=1)
            # 获取当前批次的第一个图像和预测结果
            img = x[0].cpu().numpy()
            pred = predict[0].cpu().item()
            # 绘制图像
            plt.figure(n)
            plt.imshow(img.reshape((28, 28)), cmap='gray')
            plt.title(f"True: {int(y[0])}, Pred: {int(pred)}")
            plt.axis('off')  # 不显示坐标轴
    plt.show()


def show(model, test_loader, DEVICE):
    model.eval()  # 将模型设置为评估模式
    with torch.no_grad():  # 禁用梯度计算
        fig, axes = plt.subplots(2, 3, figsize=(10, 5))  # 创建一个 1 行 2 列的子图
        axes = axes.flatten()  # 将 axes 展平为一维数组

        for n, (x, y) in enumerate(test_loader):
            if n >= 6:  # 只显示前两张图片
                break
            x = x.to(DEVICE)  # 将输入数据移到设备上
            predict = model(x).argmax(dim=1)  # 获取模型的预测结果

            # 获取当前批次的第一个图像和预测结果
            img = x[0].cpu().numpy()  # 转换为 CPU 上的 NumPy 数组
            pred = predict[0].cpu().item()  # 获取预测值

            # 绘制图像
            ax = axes[n]  # 获取子图的当前轴
            ax.imshow(img.reshape(28, 28), cmap='gray')  # 确保将图像 reshape 成 (28, 28)
            ax.set_title(f"True: {int(y[0])}, Pred: {int(pred)}")  # 标题显示真实标签和预测标签
            ax.axis('off')  # 关闭坐标轴显示

        plt.show()  # 显示所有绘制的图像


def main():
    # 定义优化器
    model = ResNet18().to(DEVICE)  # 创建模型并将模型加载到指定设备上

    # 加载模型，如果存在的话
    load_model(model, model_filename)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # 图像处理
    pipeline = transforms.Compose([
        transforms.ToTensor(),  # 将图片转换为Tensor，像素值在[0, 1]之间
        transforms.Normalize(mean=[0.1307], std=[0.3081])  # 对图像进行标准化（均值0.1307，标准差0.3081）
    ])
    # 下载
    train_set = datasets.MNIST("data", train=True, download=True, transform=pipeline)
    test_set = datasets.MNIST("data", train=False, download=True, transform=pipeline)
    # 加载 一次性加载BATCH_SIZE个打乱顺序的数据
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)

    # 如果模型没有加载（即第一次运行），进行训练
    if not os.path.exists(model_filename):
        # 训练并保存模型
        for epoch in range(1, EPOCHS + 1):  # 假设训练10个epoch
            train_model(model, DEVICE, train_loader, optimizer, criterion, epoch)
        save_model(model, model_filename)  # 训练后保存模型

    test_model(model, DEVICE, test_loader, criterion)
    show(model, test_loader, DEVICE)


if __name__ == '__main__':
    main()
