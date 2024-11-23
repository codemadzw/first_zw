import torch
import cv2
from torchvision import transforms
from ResNet import ResNet18

# 加载预训练的模型
model = ResNet18().to('cuda' if torch.cuda.is_available() else 'cpu')  # 使用 GPU 或 CPU
model.load_state_dict(torch.load('model50.pth', map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
model.eval()

# 定义图像预处理操作
transform = transforms.Compose([
    transforms.ToPILImage(),  # 转换为PIL图片格式
    transforms.Resize((28, 28)),  # 调整大小以匹配MNIST输入
    transforms.ToTensor(),  # 转换为Tensor
    transforms.Normalize((0.1307,), (0.3081,))
])


def process_image(image):
    # 转换为灰度图
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 预处理图像
    img_tensor = transform(gray_image).unsqueeze(0).to('cuda' if torch.cuda.is_available() else 'cpu')

    return img_tensor


def predict(image_tensor):
    with torch.no_grad():
        output = model(image_tensor)
        pred = output.argmax(dim=1).item()  # 获取预测的类别标签
    return pred


def process_and_predict_image(image_path):
    image = cv2.imread(image_path)
    img_tensor = process_image(image)
    pred = predict(img_tensor)

    # 在图像上绘制预测结果
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, f'Pred: {pred}', (10, 30), font, 1, (0, 255, 0), 2)

    # 显示图像
    cv2.imshow('Handwritten Digit Recognition', image)


    cv2.waitKey(0)
    cv2.destroyAllWindows()


def process_and_predict_video():
    cap = cv2.VideoCapture(0)  # 0表示默认的摄像头

    while True:
        # 获取摄像头的实时帧
        ret, frame = cap.read()
        if not ret:
            break

        # 预处理图像
        img_tensor = process_image(frame)
        pred = predict(img_tensor)

        # 在图像上绘制预测结果
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, f'Pred: {pred}', (10, 30), font, 1, (0, 255, 0), 2)

        # 显示图像
        cv2.imshow('Handwritten Digit Recognition', frame)

        # 按 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放摄像头和关闭所有窗口
    cap.release()
    cv2.destroyAllWindows()


# 主函数
if __name__ == "__main__":
    # 如果需要处理静态图片
    image_path = r"C:\Users\lucky\Desktop\number_work\test_jpgs\2.jpg"
    # process_and_predict_image(image_path)

    # 如果需要开启摄像头进行实时视频流识别
    process_and_predict_video()