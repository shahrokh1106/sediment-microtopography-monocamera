import cv2
import numpy as np #Numpy
import os
from scipy import ndimage
import math
from PIL import Image, ImageOps
from glob import glob
from os.path import join
from ntpath import basename
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import sys
from skimage import io, color, filters, measure
from scipy.ndimage import gaussian_filter
from skimage import data,exposure
from skimage.exposure import equalize_adapthist, match_histograms
from skimage.transform import warp, ProjectiveTransform

#CREStereo
import megengine as mge
import megengine.functional as F
from CREStereomaster.nets import Model
import open3d as o3d


#superglue
from pathlib import Path
import argparse
import random
import numpy as np
import matplotlib.cm as cm
import torch
from models.matching import Matching
from models.utils import (compute_pose_error, compute_epipolar_error,estimate_pose, make_matching_plot,error_colormap, AverageTimer, pose_auc, read_image,rotate_intrinsics, rotate_pose_inplane,scale_intrinsics)
from pathlib import Path
import argparse
import cv2
import matplotlib.cm as cm
import torch
from models.matching import Matching
from models.utils import (AverageTimer, VideoStreamer,make_matching_plot_fast, frame2tensor)

def preprocess (img):
    reference = cv2.imread("shell_1.png")
    image = img.astype(np.float32)
    mean, std = cv2.meanStdDev(image)
    normalized_image = (image - mean.reshape(1,3)) / std.reshape(1,3)
    normalized_image = cv2.normalize(normalized_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    equalized_img = equalize_adapthist(normalized_image)
    equalized_img = cv2.normalize(equalized_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    matched = equalized_img.copy()
    matched = match_histograms(equalized_img, reference, channel_axis=-1)
    matched = cv2.cvtColor(matched, cv2.COLOR_BGR2RGB)
    return matched


def preprocess_img (img):
    reference = cv2.imread("shell_1.png")
    image = img.astype(np.float32)
    mean, std = cv2.meanStdDev(image)
    normalized_image = (image - mean.reshape(1,3)) / std.reshape(1,3)
    normalized_image = cv2.normalize(normalized_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return normalized_image


    
def get_roi (image):
    IMAGE = image.copy()
    def draw_rectangle(event, x, y, flags, params):
        global rectangle_points, img, img_orig
        # If left mouse button is pressed down, start the crop
        if event == cv2.EVENT_LBUTTONDOWN:
            rectangle_points = [(x, y)]
        # If left mouse button is released, finish the crop
        elif event == cv2.EVENT_LBUTTONUP:
            rectangle_points.append((x, y))
            # Draw a rectangle on the image
            cv2.rectangle(img, rectangle_points[0], rectangle_points[1], (0, 255, 0), 2)
            cv2.imshow("image", img)
    # Load the image and clone it
    global img,img_orig,rectangle_points
    rectangle_points = []
    w = int(image.shape[1] * 30 / 100)
    h = int(image.shape[0] * 30 / 100)
    image = cv2.resize(image, (w, h), interpolation = cv2.INTER_AREA)
    img = image.copy()
    img_orig = img.copy()
    # Create a window and set the mouse callback function to draw_rectangle
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", draw_rectangle)
    while True:
        # Display the image
        text_0 = "Press \"r\" to reset the croping"
        text_1 = "Hold the left click and draw the box, then press \"c\" to crop"
        cv2.putText(img, text_0, (10,70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
        cv2.putText(img, text_1, (10,120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
        cv2.imshow("image", img)
        key = cv2.waitKey(1) & 0xFF
        # If 'r' is pressed, reset the cropping region
        if key == ord("r"):
            img = img_orig.copy()
        # If 'c' is pressed, break from the loop and do the cropping
        elif key == ord("c"):
            break
        elif key == ord("q"):
            break    
    cv2.destroyAllWindows()
    if len(rectangle_points)!=0:
        return [(int(rectangle_points[0][0]*100/30),int(rectangle_points[0][1]*100/30)),(int(rectangle_points[1][0]*100/30),int(rectangle_points[1][1]*100/30))]



def get_matches (left,right,max_keypoints=500,keypoint_threshold=0.001,nms_radius = 4,match_threshold = 0.1,debug=True, scale_percent=100):
    torch.set_grad_enabled(False);
    input = 0
    skip = 1
    max_length=1000000
    resize=[640, 480]
    superglue='outdoor'
    sinkhorn_iterations = 20
    
    device = 'cuda' if torch.cuda.is_available()  else 'cpu'
    config = {
            'superpoint': {
                'nms_radius': nms_radius,
                'keypoint_threshold': keypoint_threshold,
                'max_keypoints': max_keypoints
            },
            'superglue': {
                'weights': superglue,
                'sinkhorn_iterations': sinkhorn_iterations,
                'match_threshold': match_threshold,
            }
        }
    matching = Matching(config).eval().to(device)
    keys = ['keypoints', 'scores', 'descriptors']
    frame_tensor = frame2tensor(left, device)
    last_data = matching.superpoint({'image': frame_tensor})
    last_data = {k+'0': last_data[k] for k in keys}
    last_data['image0'] = frame_tensor
    last_frame = left
    last_image_id = 0

    frame_tensor = frame2tensor(right, device)
    pred = matching({**last_data, 'image1': frame_tensor})
    kpts0 = last_data['keypoints0'][0].cpu().numpy()
    kpts1 = pred['keypoints1'][0].cpu().numpy()
    matches = pred['matches0'][0].cpu().numpy()
    confidence = pred['matching_scores0'][0].cpu().numpy()

    valid = matches > -1
    mkpts0 = kpts0[valid]
    mkpts1 = kpts1[matches[valid]]
    color = cm.jet(confidence[valid])

    text = [
        'SuperGlue',
        'Keypoints: {}:{}'.format(len(kpts0), len(kpts1)),
        'Matches: {}'.format(len(mkpts0))
    ]
    k_thresh = matching.superpoint.config['keypoint_threshold']
    m_thresh = matching.superglue.config['match_threshold']
    small_text = [
        'Keypoint Threshold: {:.4f}'.format(k_thresh),
        'Match Threshold: {:.2f}'.format(m_thresh),
    ]
    show_keypoints='store_true'
    out = make_matching_plot_fast(
        last_frame, right, kpts0, kpts1, mkpts0, mkpts1, color, text,
        path=None, show_keypoints=show_keypoints, small_text=small_text)
    if debug:    
        show(out,scale_percent=scale_percent)
    return mkpts0,mkpts1

def compute_disp (left,right,model_path, n_iter, debug=True):
    def load_model(model_path):
        pretrained_dict = mge.load(model_path)
        model = Model(max_disp=256, mixed_precision=False, test_mode=True)
        model.load_state_dict(pretrained_dict["state_dict"], strict=True)
        model.eval()
        return model
    
    def inference(left, right, model, n_iter=n_iter):
        imgL = left.transpose(2, 0, 1)
        imgR = right.transpose(2, 0, 1)
        imgL = np.ascontiguousarray(imgL[None, :, :, :])
        imgR = np.ascontiguousarray(imgR[None, :, :, :])
        imgL = mge.tensor(imgL).astype("float32")
        imgR = mge.tensor(imgR).astype("float32")
        imgL_dw2 = F.nn.interpolate(imgL,size=(imgL.shape[2] // 2, imgL.shape[3] // 2),mode="bilinear",align_corners=True)
        imgR_dw2 = F.nn.interpolate(imgR,size=(imgL.shape[2] // 2, imgL.shape[3] // 2),mode="bilinear",align_corners=True)
        pred_flow_dw2 = model(imgL_dw2, imgR_dw2, iters=n_iter, flow_init=None)
        pred_flow = model(imgL, imgR, iters=n_iter, flow_init=pred_flow_dw2)
        pred_disp = F.squeeze(pred_flow[:, 0, :, :]).numpy()
        return pred_disp
    
    model_func = load_model(model_path)
    in_h, in_w = left.shape[:2]
    eval_h, eval_w = [int(e) for e in (1024,1536)]
    # show(left,scale_percent=15)
    left_img = cv2.resize(left, (eval_w, eval_h), interpolation=cv2.INTER_LINEAR)
    right_img = cv2.resize(right, (eval_w, eval_h), interpolation=cv2.INTER_LINEAR)
    pred = inference(left_img, right_img, model_func, n_iter=n_iter)
    t = float(in_w) / float(eval_w)
    disp = cv2.resize(pred, (in_w, in_h), interpolation=cv2.INTER_LINEAR) * t
    disp_vis = (disp - disp.min()) / (disp.max() - disp.min()) * 255.0
    disp_vis = disp_vis.astype("uint8")
    disp_vis = cv2.applyColorMap(disp_vis, cv2.COLORMAP_INFERNO)
    return disp,disp_vis


def compute_fundamental_matrix(points1, points2):
    '''
    Compute the fundamental matrix given the point correspondences
    
    Parameters
    ------------
    points1, points2 - array with shape [n, 3]
        corresponding points in images represented as 
        homogeneous coordinates
    '''
    # validate points
    assert points1.shape[0] == points2.shape[0], "no. of points don't match"
    
    u1 = points1[:, 0]
    v1 = points1[:, 1]
    u2 = points2[:, 0]
    v2 = points2[:, 1]
    one = np.ones_like(u1)
    
    # construct the matrix 
    # A = [u2.u1, u2.v1, u2, v2.u1, v2.v1, v2, u1, v1, 1] for all the points
    # stack columns
    A = np.c_[u1 * u2, v1 * u2, u2, u1 * v2, v1 * v2, v2, u1, v1, one]
    
    # peform svd on A and find the minimum value of |Af|
    U, S, V = np.linalg.svd(A, full_matrices=True)
    f = V[-1, :]
    F = f.reshape(3, 3) # reshape f as a matrix
    
    # constrain F
    # make rank 2 by zeroing out last singular value
    U, S, V = np.linalg.svd(F, full_matrices=True)
    S[-1] = 0 # zero out the last singular value
    F = U @ np.diag(S) @ V # recombine again
    return F

def compute_fundamental_matrix_normalized(points1, points2):
    '''
    Normalize points by calculating the centroid, subtracting 
    it from the points and scaling the points such that the distance 
    from the origin is sqrt(2)
    
    Parameters
    ------------
    points1, points2 - array with shape [n, 3]
        corresponding points in images represented as 
        homogeneous coordinates
    '''
    # validate points
    assert points1.shape[0] == points2.shape[0], "no. of points don't match"
    
    # compute centroid of points
    c1 = np.mean(points1, axis=0)
    c2 = np.mean(points2, axis=0)
    
    # compute the scaling factor
    s1 = np.sqrt(2 / np.mean(np.sum((points1 - c1) ** 2, axis=1)))
    s2 = np.sqrt(2 / np.mean(np.sum((points2 - c2) ** 2, axis=1)))
    
    # compute the normalization matrix for both the points
    T1 = np.array([
        [s1, 0, -s1 * c1[0]],
        [0, s1, -s1 * c1[1]],
        [0, 0 ,1]
    ])
    T2 = np.array([
        [s2, 0, -s2 * c2[0]],
        [0, s2, -s2 * c2[1]],
        [0, 0, 1]
    ])
    
    # normalize the points
    points1_n = T1 @ points1.T
    points2_n = T2 @ points2.T
    
    # compute the normalized fundamental matrix
    F_n = compute_fundamental_matrix(points1_n.T, points2_n.T)
    
    # de-normalize the fundamental
    return T2.T @ F_n @ T1

def compute_epipole(F):
    U, S, V = np.linalg.svd(F)
    e = V[-1, :]
    e = e / e[2]
    return e
    
def compute_matching_homographies(e2, F, shape, points1, points2):
    '''
    Compute the matching homography matrices
    '''
    h, w = shape
    # create the homography matrix H2 that moves the epipole to infinity
    
    # create the translation matrix to shift to the image center
    T = np.array([[1, 0, -w/2], [0, 1, -h/2], [0, 0, 1]])
    e2_p = T @ e2
    e2_p = e2_p / e2_p[2]
    e2x = e2_p[0]
    e2y = e2_p[1]
    # create the rotation matrix to rotate the epipole back to X axis
    if e2x >= 0:
        a = 1
    else:
        a = -1
    R1 = a * e2x / np.sqrt(e2x ** 2 + e2y ** 2)
    R2 = a * e2y / np.sqrt(e2x ** 2 + e2y ** 2)
    R = np.array([[R1, R2, 0], [-R2, R1, 0], [0, 0, 1]])
    e2_p = R @ e2_p
    x = e2_p[0]
    # create matrix to move the epipole to infinity
    G = np.array([[1, 0, 0], [0, 1, 0], [-1/x, 0, 1]])
    # create the overall transformation matrix
    H2 = np.linalg.inv(T) @ G @ R @ T

    # create the corresponding homography matrix for the other image
    e_x = np.array([[0, -e2[2], e2[1]], [e2[2], 0, -e2[0]], [-e2[1], e2[0], 0]])
    M = e_x @ F + e2.reshape(3,1) @ np.array([[1, 1, 1]])
    points1_t = H2 @ M @ points1.T
    points2_t = H2 @ points2.T
    points1_t /= points1_t[2, :]
    points2_t /= points2_t[2, :]
    b = points2_t[0, :]
    a = np.linalg.lstsq(points1_t.T, b, rcond=None)[0]
    H_A = np.array([a, [0, 1, 0], [0, 0, 1]])
    H1 = H_A @ H2 @ M
    return H1, H2
    
def get_rectified_images_opencv (left,right,pts1,pts2, debug=True, scale_percent=100):
    fundamental_matrix, inliers = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)
    pts1 = pts1[inliers.ravel() == 1]
    pts2 = pts2[inliers.ravel() == 1]
    h1, w1 = left.shape[0],left.shape[1]
    h2, w2 = right.shape[0],right.shape[1] 
    height, width = left.shape[:2]
    _, H1, H2 = cv2.stereoRectifyUncalibrated(np.float32(pts1), np.float32(pts2), fundamental_matrix, imgSize=(width, height))
    corners = np.array([[0, 0], [0, height-1], [width-1, 0], [width-1, height-1]], dtype = "float32")
    corners = np.array([corners])
    warped_corners_1 = cv2.perspectiveTransform(corners, H1)[0]
    warped_corners_2 = cv2.perspectiveTransform(corners, H2)[0]
    # get the bounding rectangle for the warped images
    min_x_1, min_y_1 = np.int0(np.min(warped_corners_1, axis=0))
    max_x_1, max_y_1 = np.int0(np.max(warped_corners_1, axis=0))
    min_x_2, min_y_2 = np.int0(np.min(warped_corners_2, axis=0))
    max_x_2, max_y_2 = np.int0(np.max(warped_corners_2, axis=0))
    # calculate the width and height of the new images
    width_1 = max_x_1 - min_x_1
    height_1 = max_y_1 - min_y_1
    width_2 = max_x_2 - min_x_2
    height_2 = max_y_2 - min_y_2
    # translation matrix to shift the images to the top left corner (minimum x and y coordinates)
    T1 = np.array([[1, 0, -min_x_1], [0, 1, -min_y_1], [0, 0, 1]], dtype = "float32")
    T2 = np.array([[1, 0, -min_x_2], [0, 1, -min_y_2], [0, 0, 1]], dtype = "float32")
    img1_rectified = cv2.warpPerspective(left,  np.dot(T1, H1), (width_1, height_1))
    img2_rectified = cv2.warpPerspective(right, np.dot(T2, H2), (width_1, height_1))
    if debug:
        show(np.hstack((img1_rectified,img2_rectified)), scale_percent=scale_percent)
    return img1_rectified,img2_rectified
    
def get_rectified_images (left,right, points1,points2, debug=True,scale_percent=100):
    F = compute_fundamental_matrix_normalized(points1, points2)
    e1 = compute_epipole(F)
    e2 = compute_epipole(F.T)
    H1, H2 = compute_matching_homographies(e2, F, left.shape[:2], points1, points2)
    h1, w1 = left.shape[0],right.shape[1]
    h2, w2 = left.shape[0],right.shape[1] 
    height, width = left.shape[:2]
    corners = np.array([[0, 0], [0, height-1], [width-1, 0], [width-1, height-1]], dtype = "float32")
    corners = np.array([corners])
    warped_corners_1 = cv2.perspectiveTransform(corners, H1)[0]
    warped_corners_2 = cv2.perspectiveTransform(corners, H2)[0]
    # get the bounding rectangle for the warped images
    min_x_1, min_y_1 = np.min(warped_corners_1, axis=0)
    min_x_1, min_y_1 = int(min_x_1), int(min_y_1)
    max_x_1, max_y_1 = np.max(warped_corners_1, axis=0)
    max_x_1, max_y_1 = int(max_x_1), int(max_y_1)
    min_x_2, min_y_2 = np.min(warped_corners_2, axis=0)
    min_x_2, min_y_2 = int(min_x_2), int(min_y_2)
    max_x_2, max_y_2 = np.max(warped_corners_2, axis=0)
    max_x_2, max_y_2 = int(max_x_2), int(max_y_2)
    # calculate the width and height of the new images
    width_1 = max_x_1 - min_x_1
    height_1 = max_y_1 - min_y_1
    width_2 = max_x_2 - min_x_2
    height_2 = max_y_2 - min_y_2
    # translation matrix to shift the images to the top left corner (minimum x and y coordinates)
    T1 = np.array([[1, 0, -min_x_1], [0, 1, -min_y_1], [0, 0, 1]], dtype = "float32")
    T2 = np.array([[1, 0, -min_x_2], [0, 1, -min_y_2], [0, 0, 1]], dtype = "float32")
    img1_rectified = cv2.warpPerspective(left.copy(),  np.dot(T1, H1), (width_1, height_1))
    img2_rectified = cv2.warpPerspective(right.copy(), np.dot(T2, H2), (width_1, height_1))
    if debug:
        show(np.hstack((img1_rectified,img2_rectified)), scale_percent=scale_percent)
    return img1_rectified,img2_rectified
    
def save_point_cloud(left,disp,Q,name, mask):
    def create_output(vertices, colors, filename):
        colors = colors.reshape(-1, 3)
        vertices = np.hstack([vertices.reshape(-1,3), colors])
        ply_header = '''ply
        format ascii 1.0
        element vertex %(vert_num)d
        property float x
        property float y
        property float z
        property uchar red
        property uchar green
        property uchar blue
        end_header
        '''
        with open(filename, 'w') as f:
            f.write(ply_header % dict(vert_num=len(vertices)))
            np.savetxt(f, vertices, '%f %f %f %d %d %d')
            
    points_3D = cv2.reprojectImageTo3D(disp, Q)
    colors = cv2.cvtColor(left, cv2.COLOR_BGR2RGB)
    mask_map=~mask.astype(bool)
    output_points = points_3D[mask_map]
    output_colors = colors[mask_map]    
    output_file = name + '.ply'     
    create_output(output_points, output_colors, output_file)
    
    cloud = o3d.io.read_point_cloud(output_file) # Read the point cloud
    cloud.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    #cloud = cloud.voxel_down_sample(voxel_size=0.02)
    #cloud, ind = cloud.remove_statistical_outlier(nb_neighbors=20,std_ratio=2.0)
    cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1,max_nn=30))
    o3d.io.write_point_cloud(name+".ply", cloud, write_ascii=True, compressed=False, print_progress=False)

def get_undistorted_roi_two_images (image_1,image_2, preprocess_flag = True, output_size = (640,480), debug= True, scale_percent = 100):
    distortion_coefficients = np.asarray([[-0.42704954,  0.2422638 , -0.00071215,  0.0004687 , -0.0872903 ]])
    camera_matrix = np.asarray([[2.82029177e+03, 0.00000000e+00, 2.05793575e+03],[0.00000000e+00, 2.82397145e+03, 1.52515669e+03],[0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    image_1=image_1[83:3074, :]
    image_2=image_2[83:3074, :]
    undist_1 = cv2.undistort(image_1, camera_matrix, distortion_coefficients, None, camera_matrix)
    undist_2 = cv2.undistort(image_2, camera_matrix, distortion_coefficients, None, camera_matrix)
    roi = get_roi (undist_1)
    undist_1 = undist_1[roi[0][1]:roi[1][1], roi[0][0]:roi[1][0]]
    undist_2 = undist_2[roi[0][1]:roi[1][1], roi[0][0]:roi[1][0]]
    undist_1 = cv2.resize(undist_1, output_size)
    undist_2 = cv2.resize(undist_2, output_size)
    
    if preprocess_flag== True:
        undist_1 = preprocess_img (undist_1)
        undist_2 = preprocess_img (undist_2)

    if debug:
        show(np.hstack((undist_1,undist_2)),scale_percent =scale_percent )
    return undist_1,undist_2


def colorCorrection(imagem, intensidade):
    resultados = [] #vetor para receber os resultados das trasnformações 
    rgb = cv2.split(imagem) #acesso a cada canal de cor
    saturacao = rgb[0].shape[0] * rgb[0].shape[1] * intensidade / 500.0 #200
    for canal in rgb:
        histograma = cv2.calcHist([canal], [0], None, [256], (0,256), accumulate=False)
        #low value
        lowvalue = np.searchsorted(np.cumsum(histograma), saturacao) #soma acumulada dos elementos valor inferior do histograma e encontra índices onde os elementos devem ser inseridos p/ ordenar
        #high value
        highvalue = 255-np.searchsorted(np.cumsum(histograma[::-1]), saturacao)#soma acumulada e sort valores superiores
        #tomar toda a informação (min/max) da curva linear para aplicar e gerar uma LUT de 256 valores a aplicar nos canais stretching
        lut = np.array([0 if i < lowvalue else (255 if i > highvalue else round(float(i-lowvalue)/float(highvalue-lowvalue)*255)) for i in np.arange(0, 256)], dtype="uint8")
        #mescla os canais de volta
        resultados.append(cv2.LUT(canal, lut))
    return cv2.merge(resultados)

def gammaCorrection(image, gamma=0.8):  #0.7
    #construir uma tabela de novos valores mapeados de pixel [0, 255] para seus valores gama ajustados
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
    for i in np.arange(0, 256)]).astype("uint8")
    #aplicar a correção 
    return cv2.LUT(image, table)

def mascaraNitidez(imagem, kernel=(5, 5), sigma=0.5, intensidade=1.0, threshold=0): #sigma 1.0, intensidade 2.0
    suavizacao = cv2.GaussianBlur(imagem, kernel, sigma)
    nitidez = float(intensidade + 1) * imagem - float(intensidade) * suavizacao
    nitidez = np.maximum(nitidez, np.zeros(nitidez.shape))
    nitidez = np.minimum(nitidez, 255 * np.ones(nitidez.shape))
    nitidez = nitidez.round().astype(np.uint8)
    if threshold > 0:
        contraste_baixo = np.absolute(imagem - suavizacao) < threshold
        np.copyto(nitidez, imagem, where=contraste_baixo)
    return nitidez

def CLAHE(imagem):
    clahe = cv2.createCLAHE(clipLimit=4, tileGridSize=(4, 4))
    for i in range(3):
        imagem[:, :, i] = clahe.apply((imagem[:, :, i]))
    return imagem

def add(a ,b):
    fusao = cv2.addWeighted(a, 0.8, b, 0.5, 0) #combina duas imagens
    #0.5/0.5 valores alfa e beta
    return fusao

def enhacement(imagem):
    brightness = 5
    contrast = 10
    img = np.int16(imagem)
    img = img * (contrast/127+1) - contrast + brightness
    img = np.clip(img, 0, 255)
    final = np.uint8(img)
    return final

def compare(a,b):
    input = a
    output= b
    cp = np.hstack((input, output))
    return cv2_imshow(cp)

def UWE(image):
    colorCorrected = colorCorrection(image, 1.0)
    gammaCorrected = gammaCorrection(colorCorrected)
    edgeEnhacement = mascaraNitidez(gammaCorrected)
    clahe = CLAHE(colorCorrected)
    gammaCorrected2 = gammaCorrection(colorCorrected)
    edgeEnhacement2 = mascaraNitidez(gammaCorrected2)
    fusao = add(edgeEnhacement, edgeEnhacement2)
    output = enhacement(fusao)
    return output

#################################################################


def show (img, scale_percent = 30, waitKey=-1):
    if scale_percent !=None:
        w = int(img.shape[1] * scale_percent / 100)
        h = int(img.shape[0] * scale_percent / 100)
        img = cv2.resize(img, (w, h), interpolation = cv2.INTER_AREA)
    cv2.imshow('image', img)
    k = cv2.waitKey(waitKey) & 0xFF
    if k == ord('s'):
        cv2.imwrite("image.png", img)
        cv2.destroyAllWindows()     
    if k == ord('q'):
        cv2.destroyAllWindows()  
    cv2.destroyAllWindows()
    


#################################################################
def make_masks(image):
    # Globals
    global finished,img,ROI,vertices    
    finished = False
    img = None
    ROI = None
    vertices = []
    def CallBackFunc(event, x, y, flags, userdata):
        global finished, img, ROI, vertices
        if event == cv2.EVENT_RBUTTONDOWN:
            if len(vertices) < 2:
                print("You need a minimum of three points!")
                return
            # Close polygon
            cv2.line(img, vertices[-1], vertices[0], (0, 0, 0), 1)

            # Mask is black with white where our ROI is
            mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
            pts = np.array([vertices], dtype=np.int32)
            cv2.fillPoly(mask, pts, 255)
            ROI = cv2.bitwise_and(img, img, mask=mask)
            finished = True
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(vertices) == 0:
                # First click - just draw point
                img[y, x] = (255, 0, 0)
            else:
                # Second, or later click, draw line to previous vertex
                cv2.line(img, (x, y), vertices[-1], (0, 0, 0), 1)
            vertices.append((x, y))
            return

    # Read image from file
    img = image.copy()
    # Check if it loaded
    if img is None:
        print("Error loading the image")
        exit(1)
    # Create a window
    cv2.namedWindow("ImageDisplay", cv2.WINDOW_NORMAL)
    # Register a mouse callback
    cv2.setMouseCallback("ImageDisplay", CallBackFunc)
    # Main loop
    while not finished:
        cv2.imshow("ImageDisplay", img)
        cv2.waitKey(50)
    return ROI


#############################################################################

def trim (input_path,output_path,scale_show=30):
    
    def trim_video(input_path, output_path, start_frame, end_frame):
        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Set the start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Write frames from the start frame to the end frame
            if cap.get(cv2.CAP_PROP_POS_FRAMES) <= end_frame:
                out.write(frame)
            else:
                break
            # cv2.imshow('Trimmed Video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        out.release()
        cv2.destroyAllWindows()
    
    # Watch the video and set the starting point by pressing 's'
    cap = cv2.VideoCapture(input_path)
    start_frame = 0
    end_frame = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        w = int(frame.shape[1] * scale_show / 100)
        h = int(frame.shape[0] * scale_show / 100)
        frame = cv2.resize(frame, (w, h), interpolation = cv2.INTER_AREA)
        text_0 = " TO TRIM THE VIDEO"
        text_1 = "Press \"s\" to set the start frame"
        text_2 = "Press \"e\" to set the end frame"
        text_3 = "Press \"q\" to exit"
        cv2.putText(frame, text_0, (0,70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
        cv2.putText(frame, text_1, (10,120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
        cv2.putText(frame, text_2, (10,170), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
        cv2.putText(frame, text_3, (10,220), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
        cv2.imshow('Trimming', frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('s'):
            start_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
            # print(f"Start Frame: {start_frame}")
            
        # Press 'e' key to set the end point
        elif key == ord('e'):
            end_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
            # print(f"End Frame: {end_frame}")
            break
        elif key == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()  
    # Trim the video based on the selected start and end frames
    trim_video(input_path,output_path,int(start_frame), int(end_frame))
    
    
##################################################################################################

def get_roi (image):
    IMAGE = image.copy()
    def draw_rectangle(event, x, y, flags, params):
        global rectangle_points, img, img_orig
        # If left mouse button is pressed down, start the crop
        if event == cv2.EVENT_LBUTTONDOWN:
            rectangle_points = [(x, y)]
        # If left mouse button is released, finish the crop
        elif event == cv2.EVENT_LBUTTONUP:
            rectangle_points.append((x, y))
            # Draw a rectangle on the image
            cv2.rectangle(img, rectangle_points[0], rectangle_points[1], (0, 255, 0), 2)
            cv2.imshow("image", img)

    # Load the image and clone it
    global img,img_orig,rectangle_points
    rectangle_points = []
    w = int(image.shape[1] * 30 / 100)
    h = int(image.shape[0] * 30 / 100)
    image = cv2.resize(image, (w, h), interpolation = cv2.INTER_AREA)
    img = image.copy()
    img_orig = img.copy()
    # Create a window and set the mouse callback function to draw_rectangle
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", draw_rectangle)
    while True:
        # Display the image
        cv2.imshow("image", img)
        key = cv2.waitKey(1) & 0xFF
        # If 'r' is pressed, reset the cropping region
        if key == ord("r"):
            img = img_orig.copy()
        # If 'c' is pressed, break from the loop and do the cropping
        elif key == ord("c"):
            break
    cv2.destroyAllWindows()
    
    return [(int(rectangle_points[0][0]*100/30),int(rectangle_points[0][1]*100/30)),(int(rectangle_points[1][0]*100/30),int(rectangle_points[1][1]*100/30))]

    