import cv2
import os
import random
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten



POS_PATH = os.path.join('data', 'pos')
NEG_PATH = os.path.join('data', 'neg')
ANC_PATH = os.path.join('data', 'anc')

os.makedirs(POS_PATH)
os.makedirs(NEG_PATH)
os.makedirs(ANC_PATH)

print(NEG_PATH)