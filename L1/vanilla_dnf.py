import cv2

import numpy as np
from scipy import signal
import scipy
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.stats import logistic

import time


def euclidean_dist(x, y):
    return np.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2)


def gaussian(distance, sigma):
    return np.exp(-(distance / sigma) ** 2 / 2)


def scale_array(arr, new_min, new_max):
    """
    Scales a 2-D NumPy array to a specified range.

    Parameters:
    arr (numpy.ndarray): A 2-D NumPy array.
    new_min (float): The new minimum value of the scaled array.
    new_max (float): The new maximum value of the scaled array.

    Returns:
    numpy.ndarray: The scaled 2-D array.
    """
    old_min = np.min(arr)
    old_max = np.max(arr)
    if old_max == old_min:
        return arr
    else:
        scaled_arr = (arr - old_min) * ((new_max - new_min) / (old_max - old_min)) + new_min
        return scaled_arr


class DNF:
    def __init__(self, width, height, initX, initY):
        self.width = width
        self.height = height
        print("Dnf size : ", width, ";", height)
        self.dt = 0.7
        self.tau = 0.65
        self.cexc = 1.25
        self.secx = 0.05 * 8
        self.cinh = 0.7 / 25
        self.sinh = 10
        self.input = np.zeros([width, height], dtype=float)
        self.potentials = np.zeros([width, height], dtype=float)
        self.lateral = np.zeros([width, height], dtype=float)
        self.kernel_size = 64
        self.kernel = np.zeros([self.kernel_size, self.kernel_size], dtype=float)
        for i in range(self.kernel_size):
            for j in range(self.kernel_size):
                d = np.sqrt(((i / self.kernel_size - 0.5) ** 2 + ((j / self.kernel_size - 0.5) ** 2))) / np.sqrt(0.5)
                self.kernel[i, j] = self.difference_of_gaussian(d)
                # self.kernel[i, j] = np.round(self.kernel[i, j] * 32) / 32
        # print(np.unique(self.kernel))
        print("init x & Y:")
        print(initX, initY)
        for i in range(8):
            for j in range(8):
                self.potentials[initY - 4 + j, initX - 4 + i] = 1
        cv2.imshow('kernel', self.kernel)
        # cv2.imshow('potentials_preview', self.potentials)

        with open('kernel.txt'.format(i), 'w') as f:
            np.savetxt(f, self.kernel, fmt='%.8e')

    def difference_of_gaussian(self, distance):
        return self.cexc * np.exp(-(distance ** 2 / (2 * self.secx ** 2))) - self.cinh * np.exp(
            -(distance ** 2 / (2 * self.sinh ** 2)))

    def optimized_DoG(self, x, y):
        return self.kernel[np.abs(x[0] - y[0]), np.abs(x[1] - y[1])]

    def gaussian_activity(self, a, b, sigma):
        for i in range(self.width):
            for j in range(self.height):
                current = (i / self.width, j / self.height)
                self.input[i, j] = gaussian(euclidean_dist(a, current), sigma) + gaussian(euclidean_dist(b, current),
                                                                                          sigma)

    def update_neuron(self, x):
        # lateral = 0
        # for i in range(self.width):
        #     for j in range(self.height):
        #         lateral += self.potentials[i, j]*self.optimized_DoG((i, j), x)
        # if self.input[x] == -1:
        #     print("before:")
        #     print(self.input[x])
        #     print(self.lateral[x])
        #     print(self.potentials[x])
        self.potentials[x] += self.dt * (-self.potentials[x] + self.lateral[x] + self.input[x]) / self.tau
        # print("after:")
        # print(self.potentials[x])
        if self.potentials[x] > 1:
            self.potentials[x] = 1
        elif self.potentials[x] < 0:
            self.potentials[x] = 0
        # print(np.amax(self.potentials))
        # print(np.amin(self.potentials))

    def update_map(self, frame_no):
        # t_before_conv = time.perf_counter()
        # with open('input.txt','w') as f:
        #     for i in self.input:
        #         np.savetxt(f, i, fmt='%.8e')
        # with open('potential_before.txt','w') as f:
        #     for i in self.potentials:
        #         np.savetxt(f, i, fmt='%.8e')
        self.lateral = signal.convolve(self.potentials, self.kernel, mode='same')
        cv2.imshow('lateral', scale_array(self.lateral, 0, 255).astype(np.uint8))
        # print("lateral:")
        # print(self.lateral)
        # if 50 < frame_no <= 70:
        #     with open('conv_large{0}.txt'.format(frame_no), 'w') as f:
        #         np.savetxt(f, self.lateral, fmt='%.8e')
        # t_after_conv = time.perf_counter()
        # with open('lateral.txt', 'w') as f:
        #     for i in self.lateral:
        #         np.savetxt(f, i, fmt='%.8e')
        # print(f"time for conv:{t_after_conv - t_before_conv:0.4f}")
        # print("potential:\n")
        # print(self.potentials.shape)
        # print("kernel:\n")
        # print(self.kernel.shape)
        # print("lateral:\n")
        # print(self.lateral.shape)
        max = np.maximum(np.max(self.lateral), np.abs(np.min(self.lateral)))
        if max != 0:
            self.lateral = self.lateral / max - 1
        # if 50 < frame_no <= 70:
        #     with open('norm_{0}.txt'.format(frame_no), 'w') as f:
        #         np.savetxt(f, self.lateral, fmt='%.8e')
        # t_norm = time.perf_counter()
        # print(f"time for norm: {t_norm - t_after_conv:0.4f}")
        # print(self.lateral)
        neurons_list = list(range(self.width * self.height))
        # np.random.shuffle(neurons_list)
        # t_before_update_neuron = time.perf_counter()
        # if 50 < frame_no <= 90:
        #     with open('./save_data04202023/img_new{0}.txt'.format(frame_no), 'w') as f:
        #         np.savetxt(f, self.input, fmt='%.8e')
        for i in neurons_list:
            self.update_neuron((i % self.width, i // self.width))
        # t_after_update_neuron = time.perf_counter()
        # print(f"time for update_neuron:{t_after_update_neuron - t_before_update_neuron:0.4f}")
        # with open('potential_after.txt','w') as f:
        #     for i in self.potentials:
        #         np.savetxt(f, i, fmt='%.8e')
        # if 50 < frame_no <= 90:
        #     with open('./save_data04202023/update_new{0}.txt'.format(frame_no), 'w') as f:
        #         np.savetxt(f, self.potentials)
        #     print(f"data_saved{frame_no:d}")


i = 0


def updatefig(*args):
    global i
    dnf.update_map()
    im.set_array(dnf.potentials)
    print(i)
    i += 1
    return im,


if __name__ == '__main__':
    fig = plt.figure()
    dnf = DNF(45, 45)
    dnf.gaussian_activity((0.1, 0.5), (0.9, 0.5), 0.1)
    im = plt.imshow(dnf.input, cmap='hot', interpolation='nearest', animated=True)
    ani = animation.FuncAnimation(fig, updatefig, interval=100, blit=True)
    plt.show()
