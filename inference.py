import numpy as np
import torch
import cv2
import C3D_model

def center_crop(frame):
    frame = frame[8:120, 30:142, :]
    return np.array(frame).astype(np.uint8)

def inference():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # 加载数据标签
    with open('./data/labels.txt', 'r') as f:
        class_names = f.readlines()
        # print(class_names)
        f.close()
    # 加载模型数据
    model = C3D_model.C3D(num_classes=101)
    checkpoint = torch.load('model_result/models/C3D_epoch-29.pth.tar', )
    model.load_state_dict(checkpoint['state_dict'])
    # 设置模式和设备
    model.to(device)
    model.eval()
    video = 'v_YoYo_g01_c01.avi'
    cap = cv2.VideoCapture(video)
    retaining = True
    clip = []
    while retaining:
        retaining, frame = cap.read() # 第一个参数为如果不是视频帧则为假从而使得停止循环
        if not retaining and frame is None:
            continue
        tmp = center_crop(cv2.resize(frame, (171, 128)))
        tmp = tmp - np.array([[[90.0, 98.0, 102.0]]])
        clip.append(tmp)
        if len(clip) == 16:
            inputs = np.array(clip).astype(np.float32)
            inputs = np.expand_dims(inputs, axis=0) # 升维，我需要的数据格式为 batch,channels,d,w,h
            inputs = np.transpose(inputs, (0, 4, 1, 2, 3))  # 转换维度位置
            inputs = torch.from_numpy(inputs)
            inputs = torch.autograd.Variable(inputs, requires_grad=False).to(device)

            with torch.no_grad():
                outputs = model.forward(inputs)

            probs = torch.nn.Softmax(dim=1)(outputs)
            label = torch.max(probs, 1)[1].detach().cpu().numpy()[0]

            cv2.putText(frame, class_names[label].split(' ')[-1]. strip(), (20, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 255), 1)
            cv2.putText(frame, "prob: %.4f" % probs[0][label],
                        (20, 40),  # 修改为 (20, 40) 表示文本左下角的坐标
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,  # 字体类型和大小
                        (0, 0, 255), 1)  # 文本颜色（BGR）和线条厚度
            # print(label)
            clip.pop(0)

        cv2.imshow('result', frame)
        cv2.waitKey(30)
    cap.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    inference()
