import os
import cv2
import numpy
from torch.utils.data import Dataset
import torch


class VideoDataset(Dataset):
    def __init__(self, dirs):
        super(VideoDataset, self).__init__()
        self.base_dir = dirs

        self.video_data=[]
        self.video_label=[]
        videos = []
        for filepath in os.listdir(self.base_dir):
            for video in os.listdir(self.base_dir + '/' + filepath):
                videos.append(self.base_dir + '/' + filepath + '/' + video)

        '''
        读取数据
        '''
        for i,video_line in enumerate(videos):
            cap = cv2.VideoCapture(video_line)
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps < 16:
                continue
            frames,label=self.get_video_and_label(video_line)
            self.video_data.append(frames)
            self.video_label.append(label)



    def __getitem__(self, index):

        frames= self.video_data[index]
        label=self.video_label[index]

        frames=torch.Tensor(frames)
        frames=frames.permute(3,0,1,2)

        return frames,label


    def __len__(self):
        return len(self.video_label)


    def get_video_and_label(self, video):
        frames = []
        label = 0 if "Non" in video.split('/')[1] else 1
        cap = cv2.VideoCapture(video)
        isOpened = cap.isOpened()
        i = 0
        while isOpened:
            if i == 16:
                # 截取前16帧的连续图片
                break
            else:
                i += 1
            flag, frame = cap.read()
            frame = cv2.resize(frame, (224, 224))
            frames.append(frame)

        return frames,label