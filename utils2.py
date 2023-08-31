import rawpy
import numpy as np
import os
import time
import cv2

def np_yuv2rgb(Y,U,V,width,height):
    IMG_WIDTH = width
    IMG_HEIGHT = height
    bgr_data = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
    V = np.repeat(V, 2, 0)
    V = np.repeat(V, 2, 1)
    U = np.repeat(U, 2, 0)
    U = np.repeat(U, 2, 1)

    c = (Y-np.array([16])) * 298
    d = U - np.array([128])
    e = V - np.array([128])

    r = (c + 409 * e + 128) // 256
    g = (c - 100 * d - 208 * e + 128) // 256
    b = (c + 516 * d + 128) // 256

    r = np.where(r < 0, 0, r)
    r = np.where(r > 255,255,r)

    g = np.where(g < 0, 0, g)
    g = np.where(g > 255,255,g)

    b = np.where(b < 0, 0, b)
    b = np.where(b > 255,255,b)

    bgr_data[:, :, 2] = r
    bgr_data[:, :, 1] = g
    bgr_data[:, :, 0] = b

    return bgr_data

def from_NV12(yuv_data,width,height):
    IMG_WIDTH = width
    IMG_HEIGHT = height
    Y_WIDTH = IMG_WIDTH
    Y_HEIGHT = IMG_HEIGHT
    Y_SIZE = int(Y_WIDTH * Y_HEIGHT)
    U_V_WIDTH = int(IMG_WIDTH / 2)
    U_V_HEIGHT = int(IMG_HEIGHT / 2)
    U_V_SIZE = int(U_V_WIDTH * U_V_HEIGHT)
    Y = np.zeros((1, IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
    U = np.zeros((1, U_V_HEIGHT, U_V_WIDTH), dtype=np.uint8)
    V = np.zeros((1, U_V_HEIGHT, U_V_WIDTH), dtype=np.uint8)

    y_start = 0
    u_v_start = y_start + Y_SIZE
    u_v_end = u_v_start + (U_V_SIZE * 2)
    Y[0, :, :] = yuv_data[y_start : u_v_start].reshape((Y_HEIGHT, Y_WIDTH))
    U_V = yuv_data[u_v_start : u_v_end].reshape((U_V_SIZE, 2))
    U[0, :, :] = U_V[:, 0].reshape((U_V_HEIGHT, U_V_WIDTH))
    V[0, :, :] = U_V[:, 1].reshape((U_V_HEIGHT, U_V_WIDTH))
    return Y, U, V

def yuv2rgb(yuv,width,height):
    IMG_WIDTH = width
    IMG_HEIGHT = height
    yuv_f = open(yuv, "rb")
    time1 = time.time()
    yuv_bytes = yuv_f.read()
    yuv_data = np.frombuffer(yuv_bytes, np.uint8)
    Y, U, V = from_NV12(yuv_data,3840,2160)

    rgb_data = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
    bgr_data = np_yuv2rgb(Y[0, :, :], U[0, :, :], V[0, :, :],3840,2160)
    return bgr_data

yuv = '/home/xly/dms/ISP_test_data/SID/sensor/long/00300_00_10s.yuv'
#out = yuv2rgb(yuv,3840,2160)
#out = out/255
#print(out)
#cv2.imwrite("frame_3.jpg", out)

