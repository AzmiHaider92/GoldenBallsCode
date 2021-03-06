
import numpy as np
import cv2
from pvsfunc import PDeinterlacer
from pvsfunc import PSourcer
from PIL import Image
import sys
import os
import pickle

original_frame_size = (720, 576)
new_frame_size = (360, 288)

def deinterlace(img):
    img = Image.fromarray(img)
    size = list(img.size)
    size[0] /= 2
    size[1] /= 2
    size = [int(x) for x in size]
    downsized = img.resize(size, Image.NEAREST)
    return np.asarray(downsized)


def dissect_and_deinterlace_video(file_full_path):
    video_name = os.path.basename(file_full_path)[:-4]
    original_video_name = video_name + ".VOB"
    capture = cv2.VideoCapture(file_full_path)
    frame_array = []
    while True:
        try:
            ret, frame = capture.read()
            if not ret:
                break
            #img = Image.open(image_path)
            #img2 = cv2.imread(image_path)
            deinterlaced_img = deinterlace(frame)
            frame_array.append(deinterlaced_img)
            #deinterlaced_img.save(os.path.join(r"C:\Users\azmihaid\OneDrive - Intel Corporation\Desktop\New folder\deinterlaced",image_path.split('\\')[-1]))
        except Exception as e:
            print(e)
            exit()
    return frame_array