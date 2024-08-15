import numpy as np
import os
import numba
import cv2

# img = r"D:\Pycharm_save\Vessel_segmentation\DRIVE_AV\training\images\0.png"
img = r"zzz.png"


@numba.jit()
def LIOT_example(img):
    '''
	This funtion is a simple example but not efficient.
	'''
    # padding 8
    img = np.asarray(img)
    gray_img = img[:, :, 1]  # convert to gray; if not retinal dataset, you can use standard grayscale api

    pad_img = np.pad(gray_img, ((8, 8)), 'constant')
    original_gray = img
    Weight = pad_img.shape[0]
    Height = pad_img.shape[1]
    Output_array = np.zeros((original_gray.shape[0], original_gray.shape[1], 4)).astype(np.uint8)  # four_direction_img
    mult = np.array([1, 2, 4, 8, 16, 32, 64, 128])  # 总和是255刚好是图像像素的最大值

    for w in range(8, Weight - 8):
        for h in range(8, Height - 8):
            orgin_value = np.array([1, 1, 1, 1, 1, 1, 1, 1]) * pad_img[w, h]
            # print(orgin_value.shape)
            Right_binary_code = orgin_value - pad_img[w + 1:w + 9, h]
            Right_binary_code[Right_binary_code > 0] = 1
            Right_binary_code[Right_binary_code <= 0] = 0
            # print("Right_binary_code", Right_binary_code)
            Left_binary_code = orgin_value - pad_img[w - 8:w, h]
            Left_binary_code[Left_binary_code > 0] = 1
            Left_binary_code[Left_binary_code <= 0] = 0
            # print("Left_binary_code", Left_binary_code)
            Up_binary_code = orgin_value - pad_img[w, h + 1:h + 9].T
            # print(pad_img[w, h + 1:h + 9].T)
            Up_binary_code[Up_binary_code > 0] = 1
            Up_binary_code[Up_binary_code <= 0] = 0
            # print("Up_binary_code", Up_binary_code)
            Down_binary_code = orgin_value - pad_img[w, h - 8:h].T
            Down_binary_code[Down_binary_code > 0] = 1
            Down_binary_code[Down_binary_code <= 0] = 0
            # print("Down_binary_code", Down_binary_code)
            Sum_Right = np.sum(mult * Right_binary_code, 0)
            # print("Sum_Right", Sum_Right)
            Sum_Left = np.sum(mult * Left_binary_code, 0)
            # print("Sum_Left", Sum_Left)
            Sum_Up = np.sum(mult * Up_binary_code, 0)
            # print("Sum_Up", Sum_Up)
            Sum_Down = np.sum(mult * Down_binary_code, 0)
            # print("Sum_Down", Sum_Down)
            Output_array[w - 8, h - 8][0] = Sum_Right
            Output_array[w - 8, h - 8][1] = Sum_Left
            Output_array[w - 8, h - 8][2] = Sum_Up
            Output_array[w - 8, h - 8][3] = Sum_Down
            # print("---" * 6)
    # print(Output_array.shape)
    return Output_array


img = cv2.imread(img)
img = cv2.resize(img, (32, 32))
# print(img)
output = LIOT_example(img)
cv2.imshow('img', img)
cv2.imshow('output', output)
cv2.imshow('output1', output[:, :, 0])
cv2.imshow('output2', output[:, :, 1])
cv2.imshow('output3', output[:, :, 2])
cv2.imshow('output4', output[:, :, 3])
cv2.waitKey(0)
