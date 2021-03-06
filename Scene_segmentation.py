import cv2
import os
import numpy as np
from skimage.metrics import structural_similarity
import matplotlib.pyplot as plt

scene_similariy_thresh = 0.56


def similarity(images):
    img1 = cv2.cvtColor(images[0], cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(images[1], cv2.COLOR_BGR2GRAY)
    # return {path.basename(images[1]): structural_similarity(img1, img2, multichannel=True)}
    return structural_similarity(img1, img2)


def calculate_image_diff(files):
    """
    calculates image similarity score
    :param files: path list of all the image (frame) files
    :return: dictionary mapping each frame to the previous frame
    """
    collection = [[files[i], files[i + 1]] for i in range(len(files) - 1)]
    Scene_starts = np.zeros(shape=(len(files),1) , dtype=bool)
    Scene_starts[0] = True
    i=1
    for two_frames in collection:
        score = similarity(two_frames)
        if score is None or score >= scene_similariy_thresh:
            Scene_starts[i] = False
        else:
            Scene_starts[i] = True
        i+=1
    return Scene_starts
    #plt.plot(scores_thresh)
    #plt.show()