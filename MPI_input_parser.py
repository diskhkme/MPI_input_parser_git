import scipy.io
import os
import cv2
import numpy as np
import h5py
from tqdm import trange
import sys
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

MPI_ROOT_PATH = 'D:/Test_Models/PoseEstim/mpi_inf_3dhp_hkkim_dataset/'
CAMERA_NUM = 5
PERSON_NUM = 1 # S1, S2
SEQUENCE_NUM = 1 # Seq1, Seq2
IMG_SOURCE_WIDTH = 2048
IMG_SOURCE_HEIGHT = 2048
IMG_RESIZE_WIDTH = 368
IMG_RESIZE_HEIGHT = 368
HEATMAP_RESIZE_WIDTH = 47
HEATMAP_RESIZE_HEIGHT = 47
HEATMAP_SIGMA = 1
NUM_JOINT = 28

hdf5_dataset = h5py.File('S{0}_Seq{1}_Camera{2}.h5'.format(PERSON_NUM,SEQUENCE_NUM,CAMERA_NUM), 'w')
hdf5_dataset.attrs.create(name="image_width", data=IMG_RESIZE_WIDTH)
hdf5_dataset.attrs.create(name="image_height", data=IMG_RESIZE_HEIGHT)
hdf5_dataset.attrs.create(name="camera_number", data=CAMERA_NUM)
hdf5_dataset.attrs.create(name="person_number", data=PERSON_NUM)
hdf5_dataset.attrs.create(name="sequence_number", data=SEQUENCE_NUM)
hdf5_dataset.attrs.create(name="heatmap_width", data=HEATMAP_RESIZE_WIDTH)
hdf5_dataset.attrs.create(name="heatmap_height", data=HEATMAP_RESIZE_HEIGHT)
hdf5_dataset.attrs.create(name="heatmap_sigma", data=HEATMAP_SIGMA)
hdf5_dataset.attrs.create(name="num_joint", data=NUM_JOINT)


annotPath = os.path.join(MPI_ROOT_PATH, 'S{0}/Seq{1}/annot.mat'.format(PERSON_NUM, SEQUENCE_NUM))
annotMat = scipy.io.loadmat(annotPath)

annot2 = annotMat['annot2'] # 2D 좌표
annot3 = annotMat['annot3'] # 3D 좌표
annot3_univ = annotMat['univ_annot3'] # Normalized 3D 좌표

joint3D = annot3[CAMERA_NUM][0] # frame x 84(3*28)
joint2D = annot2[CAMERA_NUM][0] # frame x 56(2*28)

numFrame = annot3[CAMERA_NUM][0].shape[0]

hdf5_images = hdf5_dataset.create_dataset(name='images',
                             shape=(numFrame,),
                             maxshape=(None),
                             dtype=h5py.special_dtype(vlen=np.uint8))

hdf5_3d_joint = hdf5_dataset.create_dataset(name='joint_3d',
                             shape=(numFrame,),
                             maxshape=(None),
                             dtype=h5py.special_dtype(vlen=np.float))

hdf5_2d_joint = hdf5_dataset.create_dataset(name='joint_2d',
                             shape=(numFrame,),
                             maxshape=(None),
                             dtype=h5py.special_dtype(vlen=np.float))

hdf5_2d_joint_heatmap = hdf5_dataset.create_dataset(name='joint_2d_heatmap',
                             shape=(numFrame,NUM_JOINT,),
                             maxshape=(None),
                             dtype=h5py.special_dtype(vlen=np.uint8))



# https://stackoverflow.com/questions/44945111/how-to-efficiently-compute-the-heat-map-of-two-gaussian-distribution-in-python
def GenerateHeatmap(u,v,sigma,_xres,_yres,_xlim,_ylim):

    uCoord = (u/_xlim)*_xres
    vCoord = (v/_ylim)*_yres

    m1 = (uCoord,vCoord)
    s1 = np.eye(2) * sigma
    k1 = multivariate_normal(mean=m1, cov=s1)

    # create a grid of (x,y) coordinates at which to evaluate the kernels
    xlim = (0, _xres)
    ylim = (0, _yres)
    xres = _xres
    yres = _yres

    x = np.linspace(xlim[0], xlim[1], xres)
    y = np.linspace(ylim[0], ylim[1], yres)
    xx, yy = np.meshgrid(x, y)

    # evaluate kernels at grid points
    xxyy = np.c_[xx.ravel(), yy.ravel()]
    zz = k1.pdf(xxyy)

    return zz

# 동영상
aviClipPath = os.path.join(MPI_ROOT_PATH, 'S{0}/Seq{1}/imageSequence/video_{2}.avi'.format(PERSON_NUM, SEQUENCE_NUM,CAMERA_NUM))
aviClip = cv2.VideoCapture(aviClipPath)

tr = trange(numFrame, desc='Creating HDF5 dataset', file=sys.stdout)

# 각 프레임별로 데이터 저장 진행
for i in tr:
    # 동영상 저장
    ret, frame = aviClip.read()
    frame = cv2.resize(frame,dsize=(IMG_RESIZE_HEIGHT,IMG_RESIZE_WIDTH))
    hdf5_images[i] = frame.reshape(-1)

    # Joint 3D 저장
    hdf5_3d_joint[i] = joint3D[i,:].copy()

    # Joint 2D 저장
    hdf5_2d_joint[i] = joint2D[i,:].copy()

    # Heatmap 생성 및 저장
    stacked_heatmap = np.empty(())
    joint2D_arr = joint2D[i,:].reshape((28,2))
    jointIndex = 0
    for u,v in joint2D_arr:
        per_joint_heatmap = GenerateHeatmap(u,v,HEATMAP_SIGMA,
                                                HEATMAP_RESIZE_WIDTH, HEATMAP_RESIZE_HEIGHT,
                                                IMG_SOURCE_WIDTH, IMG_SOURCE_HEIGHT)
        per_joint_heatmap = per_joint_heatmap*255
        heatmap = per_joint_heatmap.astype(np.uint8)
        hdf5_2d_joint_heatmap[i,jointIndex] = heatmap
        jointIndex = jointIndex+1

hdf5_dataset.close()
aviClip.release()