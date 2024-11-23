# 导入PyTorch和其他相关的库
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
# 全连接神经网络（Feedforward Neural Network, FNN）
# 设置超参数
learning_rate = 0.01  # 学习率
batch_size = 64  # 批次大小
num_epochs = 1  # 迭代次数

# 加载MNIST数据集
train_data = torchvision.datasets.MNIST(root='./data', train=True, transform=torchvision.transforms.ToTensor(),
                                        download=True)
test_data = torchvision.datasets.MNIST(root='./data', train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)

# 创建数据加载器
train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)


# 定义神经网络模型
class NeuralNet(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.input_size = input_size  # 输入层大小，即图像的像素数
        self.hidden_size = hidden_size  # 隐藏层大小
        self.num_classes = num_classes  # 输出层大小，即类别数
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)  # 定义第一层全连接层
        self.relu = torch.nn.ReLU()  # 定义ReLU激活函数
        self.fc2 = torch.nn.Linear(self.hidden_size, self.num_classes)  # 定义第二层全连接层

    def forward(self, x):
        out = x.view(-1, self.input_size)  # 将输入图像展平为一维向量
        out = self.fc1(out)  # 通过第一层全连接层
        out = self.relu(out)  # 通过ReLU激活函数
        out = self.fc2(out)  # 通过第二层全连接层
        return out


# 实例化模型，损失函数和优化器
model = NeuralNet(28 * 28, 500, 10)  # 输入层为28*28=784，隐藏层为500，输出层为10
criterion = torch.nn.CrossEntropyLoss()  # 定义交叉熵损失函数
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  # 定义随机梯度下降优化器


# 训练函数
def train():
    total_step = len(train_loader)  # 计算总的批次数
    for epoch in range(num_epochs):  # 对每一轮迭代
        correct = 0  # 记录正确预测的个数
        total = 0  # 记录总的样本数
        for i, (images, labels) in enumerate(train_loader):  # 对每一个批次
            images = images.reshape(-1, 28 * 28)  # 将图像展平为一维向量
            outputs = model(images)  # 计算模型的输出
            loss = criterion(outputs, labels)  # 计算损失
            optimizer.zero_grad()  # 清空梯度
            loss.backward()  # 反向传播计算梯度
            optimizer.step()  # 更新权重

            _, predicted = torch.max(outputs.data, 1)  # 获取预测的类别
            total += labels.size(0)  # 累加样本数
            correct += (predicted == labels).sum()  # 累加正确预测的个数

            if (i + 1) % 100 == 0:  # 每100个批次打印一次结果
                print('Epoch {}, Step {}/{}, Loss: {:.4f}, Accuracy: {:.2f}%'
                      .format(epoch + 1, i + 1, total_step, loss.item(), 100 * correct / total))
        print('Epoch {}, Loss: {:.4f}, Accuracy: {:.2f}%'
              .format(epoch + 1, loss.item(), 100 * correct / total))  # 每一轮迭代打印一次结果


# 测试函数
def test():
    with torch.no_grad():  # 不计算梯度
        correct = 0  # 记录正确预测的个数
        total = 0  # 记录总的样本数
        for images, labels in test_loader:  # 对每一个批次
            images = images.reshape(-1, 28 * 28)  # 将图像展平为一维向量
            outputs = model(images)  # 计算模型的输出
            _, predicted = torch.max(outputs.data, 1)  # 获取预测的类别
            total += labels.size(0)  # 累加样本数
            correct += (predicted == labels).sum()  # 累加正确预测的个数

        print('Test Accuracy: {:.2f}%'.format(100 * correct / total))  # 打印测试结果


# 调用训练函数和测试函数
train()
test()

# 保存模型权重
torch.save(model.state_dict(), 'model_number.ckpt')


# 预测函数，输入一张图像，输出预测的数字
def predict(image):
    image = image.reshape(-1, 28 * 28)  # 将图像展平为一维向量
    output = model(image)  # 计算模型的输出
    _, predicted = torch.max(output.data, 1)  # 获取预测的类别
    return predicted.item()  # 返回预测的数字


# 加载模型权重
model.load_state_dict(torch.load('model_number.ckpt'))


# 随机选择四张测试图像进行预测，并显示图像和结果
num_images = 4  # 指定要显示的图像数量
plt.figure(figsize=(10, 10))  # 设置图形窗口的大小
for i in range(num_images):
    index = np.random.randint(0, len(test_data))  # 随机生成一个索引值
    image, label = test_data[index]  # 获取对应的图像和标签
    prediction = predict(image)  # 调用预测函数，得到预测结果
    plt.subplot(2, 2, i + 1)  # 设置子图的位置
    plt.imshow(image[0], cmap='gray')  # 显示图像
    plt.title('Label: {}, Prediction: {}'.format(label, prediction))  # 显示标签和预测结果
plt.show()  # 显示图形窗口

