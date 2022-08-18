import numpy as np
import os
import cv2
import torch
from train import train_model
from datalist import VideoDataset
from torch.utils.data import DataLoader
from test import test
from eval import eval


if __name__ == '__main__':
    train_model()
    #test()
    #eval()