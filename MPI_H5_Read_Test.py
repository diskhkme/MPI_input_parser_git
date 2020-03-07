import scipy.io
import os
import cv2
import numpy as np
import h5py
from tqdm import trange
import sys
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

HDF5_FILE_PATH = 'S1_Seq1_Camera0.h5'

with h5py.File(HDF5_FILE_PATH,'r') as f:
    images = f['images']
    joint3ds = f['joint_3d']
    joint2ds = f['joint_2d']
    heatmaps = f['joint_2d_heatmap']

    num_frames = images.shape[0]
    img_width = f.attrs['image_width']
    img_height = f.attrs['image_height']
    num_joints = f.attrs['num_joint']
    heatmap_width = f.attrs['heatmap_width']
    heatmap_height = f.attrs['heatmap_height']

    target_frame = 0 #확인해볼 프레임

    fig = plt.figure(figsize=(10,10))
    fig_heatmap = plt.figure(figsize=(10,10))


    #------------------Image Plot--------------------#
    frame_image = images[target_frame]
    frame_image = frame_image.reshape((img_width,img_height,-1))

    ax1=fig.add_subplot(2,2,1)
    ax1.imshow(frame_image)

    # ------------------3D Joint Plot--------------------#
    joint3d = joint3ds[target_frame]
    joint3d = joint3d.reshape((num_joints,-1))

    X = joint3d[:,0]
    Y = joint3d[:,1]
    Z = joint3d[:,2]
    ax2 = fig.add_subplot(2,2,2,projection='3d')
    ax2.scatter(X,Y,Z)

    # Create cubic bounding box to simulate equal aspect ratio
    max_range = np.array([X.max() - X.min(), Y.max() - Y.min(), Z.max() - Z.min()]).max()
    Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (X.max() + X.min())
    Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (Y.max() + Y.min())
    Zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + 0.5 * (Z.max() + Z.min())
    # Comment or uncomment following both lines to test the fake bounding box:
    for xb, yb, zb in zip(Xb, Yb, Zb):
        ax2.plot([xb], [yb], [zb], 'w')

    # ------------------2D Joint Plot--------------------#
    joint2d = joint2ds[target_frame]
    joint2d = joint2d.reshape((num_joints, -1))

    u = joint2d[:,0]
    v = joint2d[:,1]
    ax3 = fig.add_subplot(2,2,3)
    ax3.scatter(u,v)
    ax3.axis('equal')


    # ------------------Heatmap Plot--------------------#
    heatmap = heatmaps[target_frame]

    entireJointHeatMap = np.zeros((heatmap_width,heatmap_height))
    for i in range(num_joints):
        currentJointMap = heatmap[i]
        currentJointMap = currentJointMap.reshape((heatmap_width,heatmap_height))
        entireJointHeatMap = entireJointHeatMap + currentJointMap

        ax = fig_heatmap.add_subplot(7, 4, i+1)
        ax.imshow(currentJointMap)

    entireJointHeatMap = (entireJointHeatMap/entireJointHeatMap.max()) * 255

    ax4 = fig.add_subplot(2,2,4)
    ax4.imshow(entireJointHeatMap)



    plt.show()
