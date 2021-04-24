import sys
import cv2
import os
from tqdm import tqdm
from glob import glob
import warnings

def dissect_video(file_path, dump_folder):
    # if len(sys.argv) == 1:
    #     file_path = input(
    #         "please enter the path to the video to be dismembered:\n")
    # else:
    #     file_path = sys.argv[1]
    if glob(f'{dump_folder}/*.png'):
        warnings.warn('Video already dissected!')
        return
    capture = cv2.VideoCapture(file_path)
    fps = capture.get(cv2.CAP_PROP_FPS)
    try:
        video_name = os.path.basename(file_path)[:-4]

        # dump_dir = os.path.join('.', os.path.basename(os.path.dirname(file_path)), video_name + '_dump')
        dump_dir = dump_folder
        os.makedirs(dump_dir, exist_ok=True)
        dump_path = dump_dir + '/{:05}.png'
        print(file_path, dump_dir, dump_path)

        idx = 0
        frames = 0
        # while capture.grab():
        #     frames += 1
        # capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        # progress_bar = tqdm(total=int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        progress_bar = tqdm(total=10000)
        while True:
            try:
                ret, frame = capture.read()
                if not ret:
                    break
                cv2.imwrite(dump_path.format(idx), frame)
                idx += 1
                if not idx % 500:
                    if idx >= progress_bar.total:
                        progress_bar.total += 2000
                    progress_bar.update(idx)
            except Exception as e:
                print(e)
                exit()
    finally:
        capture.release()
    return dump_dir

if __name__ == '__main__':
    dissect_video(r"C:\Users\azmihaid\OneDrive - Intel Corporation\Desktop\New folder\1.01\VTS_01_1.VOB", r"C:\Users\azmihaid\OneDrive - Intel Corporation\Desktop\New folder\1.01\work\1.07_obv")