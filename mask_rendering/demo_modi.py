## A simple demostration of how to run

# before import, make sure FaceBoxes and Sim3DR are built successfully, e.g.,
# sh build.sh

import cv2
import yaml

import argparse

from FaceBoxes import FaceBoxes
from TDDFA import TDDFA
from utils.functions import draw_landmarks
from utils.render import render
from utils.depth import depth

import matplotlib.pyplot as plt


def mask3d_render(path_str = 'examples/inputs/emma.jpg'):

    # Load configs
    cfg = yaml.load(open('configs/mb1_120x120.yml'), Loader=yaml.SafeLoader)


    # Init FaceBoxes and TDDFA, recommend using onnx flag
    onnx_flag = True  # or True to use ONNX to speed up
    if onnx_flag:
        import os
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
        os.environ['OMP_NUM_THREADS'] = '4'

        from FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX
        from TDDFA_ONNX import TDDFA_ONNX

        face_boxes = FaceBoxes_ONNX()
        tddfa = TDDFA_ONNX(**cfg)
    else:
        tddfa = TDDFA(gpu_mode=False, **cfg)
        face_boxes = FaceBoxes()


    # given an image path
    #img_fp = 'examples/inputs/emma.jpg'
    img_fp = path_str
    print(img_fp)
    img = cv2.imread(img_fp)
    #plt.imshow(img[..., ::-1])


    # Detect faces using FaceBoxes
    boxes = face_boxes(img)
    print(f'Detect {len(boxes)} faces')
    print(boxes)


    # Regressing 3DMM parameters, reconstruction and visualization
    param_lst, roi_box_lst = tddfa(img, boxes)

    # reconstruct vertices and visualizing sparse landmarks
    #dense_flag = False
    #ver_lst = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)
    #draw_landmarks(img, ver_lst, dense_flag=dense_flag)

    # reconstruct vertices and visualizing dense landmarks
    #dense_flag = True
    #ver_lst = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)
    #draw_landmarks(img, ver_lst, dense_flag=dense_flag)

    # reconstruct vertices and render
    ver_lst = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=True)
    print('finished recon_vers')
    res = render(img, ver_lst, tddfa.tri, alpha=0.6, wfp='result02.png', show_flag=False);
    print('finished render')
    
    return res

    # reconstruct vertices and render depth
    #ver_lst = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)
    #depth(img, ver_lst, tddfa.tri, show_flag=True);
    
#if __name__ == '__main__':
    #parser = argparse.ArgumentParser(description='The demo of still image of 3DDFA_V2')
    #parser.add_argument('-c', '--config', type=str, default='configs/mb1_120x120.yml')
    #parser.add_argument('-f', '--img_fp', type=str, default='examples/inputs/trump_hillary.jpg')
    #parser.add_argument('-m', '--mode', type=str, default='cpu', help='gpu or cpu mode')
    #parser.add_argument('-o', '--opt', type=str, default='2d_sparse',
    #                    choices=['2d_sparse', '2d_dense', '3d', 'depth', 'pncc', 'uv_tex', 'pose', 'ply', 'obj'])
    #parser.add_argument('--show_flag', type=str2bool, default='false', help='whether to show the visualization result')
    #parser.add_argument('--onnx', action='store_true', default=True)

    #args = parser.parse_args()
    #main(args)