import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import *
import h5py
import matplotlib.pyplot as plt

if __name__ == '__main__':
    filename = r'C:\Users\azmihaid\Downloads\X_test.h5'
    with h5py.File(filename, "r") as f:
        # List all groups
        print("Keys: %s" % f.keys())
        a_group_key = list(f.keys())[0]

        # Get the data
        data = list(f[a_group_key])
    end = True