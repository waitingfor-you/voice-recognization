import os
from sklearn.model_selection import train_test_split
import cv2
import numpy as np

def process_video(ori_data_path, video, action_name, save_dir):
    resize_height = 128
    resize_width = 171
    video_filename = video.split('.')[0]
    if not os.path.exists(os.path.join(save_dir, video_filename)):
        os.mkdir(os.path.join(os.path.join(save_dir, video_filename)))

    capture = cv2.VideoCapture(os.path.join(ori_data_path, action_name, video))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))

    EXTRACT_FREQUENCY = 4
    if frame_count // EXTRACT_FREQUENCY <= 16:
        EXTRACT_FREQUENCY -= 1
        if frame_count // EXTRACT_FREQUENCY <= 16:
            EXTRACT_FREQUENCY -= 1
            if frame_count // EXTRACT_FREQUENCY <= 16:
                EXTRACT_FREQUENCY -= 1

    count = 0
    i = 0
    retaining = True

    while(count < frame_count and retaining):
        retaining, frame = capture.read()
        if frame is None:
            continue

        if count % EXTRACT_FREQUENCY == 0:
            if(frame_height != resize_height) or (frame_width != resize_width):
                frame = cv2.resize(frame, (resize_width,resize_height))
            cv2.imwrite(filename=os.path.join(save_dir, video_filename, '000{}.jpg'.format(str(i))), img=frame)
            i += 1
        count += 1
    capture.release()

def preprocess(ori_data_path, out_data_path, file):

    if not os.path.exists(out_data_path):
        os.mkdir(out_data_path)
        os.mkdir(os.path.join(out_data_path, 'train'))
        os.mkdir(os.path.join(out_data_path, 'val'))
        os.mkdir(os.path.join(out_data_path, 'test'))

    assert os.path.exists(os.path.join(ori_data_path, file)), "路径不存在！！！"
    file_path = os.path.join(ori_data_path, file)
    # 这样子就有了训练的数据集位置
    video_files = [name for name in os.listdir(file_path)]
    train_and_valid, test = train_test_split(video_files, test_size=0.2, random_state=42)
    train, val = train_test_split(train_and_valid, test_size=0.2, random_state=42)
    # 实现对数据集的划分
    train_dir = os.path.join(out_data_path, 'train', file)
    val_dir = os.path.join(out_data_path, 'val', file)
    test_dir = os.path.join(out_data_path, 'test', file)
    if not os.path.exists(train_dir):
        os.mkdir(train_dir)
    if not os.path.exists(val_dir):
        os.mkdir(val_dir)
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)

    for video in train:
        process_video(ori_data_path, video, file, train_dir)
    for video in val:
        process_video(ori_data_path, video, file, val_dir)
    for video in test:
        process_video(ori_data_path, video, file, test_dir)
    print('{}划分完成'.format(file))

if __name__ == '__main__':
    ori_data_path = r'C:\Users\DELL\Desktop\yu\UCF-101'
    out_data_path = 'data/ufc101'

    preprocess(ori_data_path, out_data_path, 'Skiing')