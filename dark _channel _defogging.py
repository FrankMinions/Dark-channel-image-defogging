'''
    author:@Liu
'''
import cv2 as cv
import numpy as np
import math
import matplotlib.pyplot as plt

image = cv.imread("./images/shanghai.jpg")
m_raw, n_raw = image.shape[:2]

def dark_channel_gray(mode, input):
    '''If mode is "a", the global atmospheric light intensity A can be solved.
    If mode is not "a",then the transmittance t can be solved.Here we refer
    to the two solutions defined by the dark channel.
    '''

    if mode == "a":
        gray = np.zeros((input.shape)).astype('uint8')
        for m in range(input.shape[0]):
            for n in range(input.shape[1]):
                gray[m, n, :] = min(input[m, n, :])
        return gray
    else:
        gray = np.zeros((input.shape)).astype('float64')
        for m in range(input.shape[0]):
            for n in range(input.shape[1]):
                gray[m, n, :] = min(input[m, n, :])
        return gray

def min_filter(input, kernel_size, stride):
    input_copy = input.copy()
    m, n = input.shape[:2]
    for i in range(0, m, stride):
        if i + kernel_size < m:
            for j in range(0, n, stride):
                if j + kernel_size < n:
                    input_copy[i + math.floor(kernel_size / 2), j + math.floor(kernel_size / 2), :] = np.min(input[i:i+kernel_size, j:j+kernel_size, :])

    return input_copy


gray = dark_channel_gray('a', image)
dark_channel = min_filter(gray, 15, 1)
pixel_num = math.floor((gray.shape[0] * gray.shape[1]) * 0.001)

m_dark_c, n_dark_c = dark_channel.shape[:2]
gray_hist = {}
for k in range(256):
    gray_hist.setdefault(k, 0)
for i in range(m_dark_c):
    for j in range(n_dark_c):
        gray_hist[dark_channel[i, j, 0]] += 1

s, index = 0.0, 0
gray_order = list(gray_hist.keys())
gray_num = list(gray_hist.values())
for i in range(1, len(gray_num)):
    s += gray_num[-i]
    if s > pixel_num:
        index = i
        break

pixel_grate = gray_order[-index]

flag = False
A_index = []
for i in range(m_dark_c):
    for j in range(n_dark_c):
        if dark_channel[i, j, 0] >= pixel_grate:
            A_index.append([i, j])
            if len(A_index) > pixel_num:
                flag = True
                break
    if flag == True:
        break

sum_r, sum_g, sum_b = 0.0, 0.0, 0.0
for l in range(len(A_index)):
    sum_b += image[A_index[l][0], A_index[l][1], 0]
    sum_g += image[A_index[l][0], A_index[l][1], 1]
    sum_r += image[A_index[l][0], A_index[l][1], 2]

# solving A: global atmospheric light intensity
A_b, A_g, A_r = sum_b / len(A_index), sum_g / len(A_index), sum_r / len(A_index)

image_A = image.copy()
image_A = image_A.astype('float64')
image_A[:, :, 0] = image_A[:, :, 0] / A_b
image_A[:, :, 1] = image_A[:, :, 1] / A_g
image_A[:, :, 2] = image_A[:, :, 2] / A_r

gray_A = dark_channel_gray('t', image_A)
dark_channel_A = min_filter(gray_A, 15, 1)
t = 1 - 0.95 * dark_channel_A   # solving transmittance

t_temp = t * 255
t_temp = t_temp.astype('uint8')

t_guided_filter = cv.ximgproc.guidedFilter(image, t_temp, 180, 1e-6)   # guided  filter
t_guided_filter = t_guided_filter / 255.

J_image = image.copy()

for i in range(m_raw):
    for j in range(n_raw):
        J_image[i, j, 0] = (image[i, j, 0] - A_b) / max(t_guided_filter[i, j, 0], 0.1) + A_b
        J_image[i, j, 1] = (image[i, j, 1] - A_g) / max(t_guided_filter[i, j, 1], 0.1) + A_g
        J_image[i, j, 2] = (image[i, j, 2] - A_r) / max(t_guided_filter[i, j, 2], 0.1) + A_r

J_image = cv.blur(J_image, ksize=(3, 3))   # optional

fig = plt.figure()

ax1 = fig.add_subplot(221)
plt.imshow(image[:, :, ::-1])
plt.title('Raw Image')
plt.xticks([])
plt.yticks([])

ax2 = fig.add_subplot(222)
plt.imshow(J_image[:, :, ::-1])
plt.title('Defogging Image($\omega$=0.95)')
plt.xticks([])
plt.yticks([])

ax3 = fig.add_subplot(223)
plt.imshow(dark_channel_A)
plt.title('Dark Channel')
plt.xticks([])
plt.yticks([])

ax4 = fig.add_subplot(224)
plt.imshow(t_guided_filter)
plt.title('Coarse Transmittance Diagram')
plt.xticks([])
plt.yticks([])

plt.show()
