import os
import cv2
import numpy as np
import tensorflow as tf

def resize_image(img, resize_shape, interpolation=cv2.INTER_NEAREST):
    return cv2.resize(img, resize_shape, interpolation=interpolation)

def grayscale_image(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def normalize_image(img):
    return img / 255.0

def crop_mask(img, mask_value):
    binary_image = cv2.inRange(img, mask_value, mask_value)
    return binary_image