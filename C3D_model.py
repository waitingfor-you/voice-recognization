import torch
import torch.nn as nn

class C3D(nn.Module):
    def __init__(self, num_classes,prtrained=True):
        super(C3D, self).__init__()
        self.pretrained = prtrained
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)) # 第一个卷积层
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)) # 第一个池化层

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))  # 第二个卷积层
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))  # 第二个池化层

        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))  # 第三个卷积层
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))  # 第三个卷积层
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))  # 第三个池化层

        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))  # 第四个卷积层
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))  # 第四个卷积层
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))  # 第四个池化层

        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))  # 第五个卷积层
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))  # 第五个卷积层
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))  # 第五个池化层

        self.fc6 = nn.Linear(8192, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, num_classes)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)  # 随机失活，也就是使得有些参数不更新
        self.__init_weight()
        if self.pretrained:
            self.__load__pretrained_weights()

    def forward(self, x):
        # 数据格式为 channels * flash * h * w
        x = self.relu(self.conv1(x))
        x = self.pool1(x)

        x = self.relu(self.conv2(x))
        x = self.pool2(x)

        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool3(x)

        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        x = self.pool4(x)

        x = self.relu(self.conv5a(x))
        x = self.relu(self.conv5b(x))
        x = self.pool5(x)

        x = x.reshape(-1, 8192)  # 原来x是一个四维的，所以需要我们把他转换成一维的，结果为8192，1
        x = self.relu(self.fc6(x))
        x = self.dropout(x)
        x = self.relu(self.fc7(x))
        x = self.dropout(x)
        x = self.fc8(x)

        return x

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):  # 检测是否都是3D的，如果是则手动初始化防止初始化过于随机
                torch.nn.init.kaiming_normal(m.weight)

    def __load__pretrained_weights(self):
        corresp_name = {
            'features.0.weight': 'conv1.weight',
            'features.0.bias': 'conv1.bias',

            'features.3.weight': 'conv2.weight',
            'features.3.bias': 'conv2.bias',

            'features.6.weight': 'conv3a.weight',
            'features.6.bias': 'conv3a.bias',

            'features.8.weight': 'conv3b.weight',
            'features.8.bias': 'conv3b.bias',

            'features.11.weight': 'conv4a.weight',
            'features.11.bias': 'conv4a.bias',

            'features.13.weight': 'conv4b.weight',
            'features.13.bias': 'conv4b.bias',

            'features.16.weight': 'conv5a.weight',
            'features.16.bias': 'conv5a.bias',

            'features.18.weight': 'conv5b.weight',
            'features.18.bias': 'conv5b.bias',

            'classifier.0.weight': 'fc6.weight',
            'classifier.0.bias': 'fc6.bias',

            'classifier.3.weight': 'fc7.weight',
            'classifier.3.bias': 'fc7.bias',
        }
        p_dict = torch.load('ucf101-caffe.pth')
        s_dict = self.state_dict()  # 这段是显示自己的初始化参数，使用的是kaimin初始法
        # print(s_dict['conv1.weight'])
        # 循环替换模型参数
        for name in p_dict:

            if name not in corresp_name:
                continue
            s_dict[corresp_name[name]] = p_dict[name]
        self.load_state_dict(s_dict)


if __name__ == '__main__':
    from torchsummary import summary

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # inputs = torch.rand(1, 3, 16, 112, 112) # batch，channel， flash， w， h

    net = C3D(num_classes=101).to(device)

    # print(summary(net, (3, 16, 112, 112)))