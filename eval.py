import torch
import cv2
from model import C3D


def eval():
    model = C3D()
    model.load_state_dict(torch.load('parameter8.pth', map_location=torch.device('cpu')), False)
    model.eval()
    'v/NonViolence/2021-12-09-19-08-31.mp4'
    video = 'v/NonViolence/2021-12-09-19-08-31.mp4'
    frames = []
    cap = cv2.VideoCapture(video)
    isOpened = cap.isOpened()
    d = 2 #间隔
    i = 0
    while isOpened:
        flag, frame = cap.read()
        if i%d == 0:
            frame = cv2.resize(frame, (224, 224))
            frames.append(frame)
        if i == 16*d - 1:
            # 截取16帧的图片
            break
        else:
            i += 1
    frames = torch.tensor(frames).float()
    frames=frames.permute(3,0,1,2)
    frames = torch.unsqueeze(frames, 0)
    out = model(frames)
    print(out)