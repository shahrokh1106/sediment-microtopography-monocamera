"""
Main file
Author: Mihailo Azhar

If used please cite:
Azhar, Mihailo, et al. "An RGB-D framework for capturing soft‐sediment microtopography." 
Methods in Ecology and Evolution 13.8 (2022): 1730-1745.
"""

import numpy as np
import scipy
import math
from scipy.stats import kurtosis, skew


def get_orientation(points_3D, target_plane):
    mean_point = np.mean(points_3D,axis=0)
    R = points_3D - mean_point
    #D, V = scipy.linalg.eig(np.matmul(R.T, R))
    D, V = scipy.linalg.eigh(np.matmul(R.T, R))
    n = V[:,0]
    V = V[:,1:]
    angle = math.degrees(math.acos(np.dot(n,target_plane)))
    if angle > 90:
        angle = 180-angle
    return angle, n

def rotate_x_axis(points, angle, inverse=False):
    angle_rad = math.radians(angle)
    rx = np.array([[1, 0, 0],[0,math.cos(angle_rad),-math.sin(angle_rad)], [0, math.sin(angle_rad), math.cos(angle_rad)]])
    if inverse: 
        r_points_3D = np.matmul(rx.T, points)
    else:
        r_points_3D = np.matmul(rx, points)
    return r_points_3D

def rotate_y_axis(points, angle, inverse=False):
    angle_rad = math.radians(angle)
    ry = np.array([[math.cos(angle_rad), 0, math.sin(angle_rad)],[0, 1, 0], [-math.sin(angle_rad), 0, math.cos(angle_rad)]])

    if inverse:
        r_points_3D = np.matmul(ry.T, points)
    else:
        r_points_3D = np.matmul(ry, points)
    return r_points_3D

def reorient_surface(points_3D, im_og, target_plane):    
    angle_change = []
    angle, n = get_orientation(points_3D,target_plane)
    angle_change.append(angle)
    #angle_rad = math.radians(angle)
    
    #rx = np.array([[1, 0, 0],[0,math.cos(angle_rad),-math.sin(angle_rad)], [0, math.sin(angle_rad), math.cos(angle_rad)]])
    #ry = np.array([[math.cos(angle_rad), 0, math.sin(angle_rad)],[0, 1, 0], [-math.sin(angle_rad), 0, math.cos(angle_rad)]])
    #r_array = (ry * full3d')';
    #r_points_3D = np.matmul(ry, points_3D.T)
    #r_points_3D = np.matmul(rx.T, points_3D.T)

    # First Rotation (we rotate in x to start)
    r_points_3D = rotate_x_axis(points_3D.T, angle, True)
    im_og = rotate_x_axis(im_og.T, angle, True)
    angle, n = get_orientation(r_points_3D.T,target_plane)
    angle_change.append(angle)
    n = n * -1  ## ADDED TO MATCH MATLAB script

    # Second rotation
    #angle_rad = math.radians(angle)
    #ry = np.array([[math.cos(angle_rad), 0, math.sin(angle_rad)],[0, 1, 0], [-math.sin(angle_rad), 0, math.cos(angle_rad)]])
    #rx = np.array([[1, 0, 0],[0,math.cos(angle_rad),-math.sin(angle_rad)], [0, math.sin(angle_rad), math.cos(angle_rad)]])
    if np.sign(n[0]) == 1:
        #r_points_3D_2 = np.matmul(ry.T, r_points_3D)
        r_points_3D_2 = rotate_y_axis(r_points_3D, angle, True)
        im_og = rotate_y_axis(im_og, angle, True)
    elif np.sign(n[0]) == -1:
        #r_points_3D_2 = np.matmul(ry, r_points_3D)
        r_points_3D_2 = rotate_y_axis(r_points_3D, angle)
        im_og = rotate_y_axis(im_og, angle)
    else:
        #r_points_3D_2 = np.matmul(ry, r_points_3D)
        r_points_3D_2 = rotate_y_axis(r_points_3D, angle)
        im_og = rotate_y_axis(im_og, angle)
    
    angle, n = get_orientation(r_points_3D_2.T,target_plane)
    angle_change.append(angle)
    
    ## TODO: Select nth rotation minimises the angle to fitted plane/quad
    
    return r_points_3D_2, im_og, angle_change

def get_abbott_curve(detrended_surface,imageHeight, imageWidth, Nl, Nt):
    zl = np.linspace(np.min(detrended_surface), np.max(detrended_surface),200)
    num_points = detrended_surface.shape[0]
    perc = np.zeros(np.shape(zl))
    for i in range(0,len(zl)):
        # This is how it is done in matlab but btwarea isn't simple pixel counting
        #perc[i] = 100 * (bwarea(z > zl[i]) / (imageHeight * imageWidth)) 
        #perc[i] = 100 * (np.sum(detrended_surface > zl[i])/(imageHeight * imageWidth))
        perc[i] = 100 * (np.sum(detrended_surface > zl[i])/(num_points))
   
    # value of z at Nl percent of data
    lower_ind = np.where(perc - Nl <= 0)[0][0]# , 1 , 'first')
    top_ind = np.where(perc - Nt <= 0)[0][0]# , 1 , 'first')
    zn_lower = zl[lower_ind]
    zn_top = zl[top_ind]
    return (zn_lower, zn_top, lower_ind, top_ind, zl, perc)

def get_abbott_stats(detrended_surface, imageHeight, imageWidth):
    ## NOTE: 15th April 2022 - changed positive vals to count < 0 vals 
    ##       values >0 (positive) are depressions/holes
    ##       values <0 (negative) are crests/mounds 
    #detrended_surface = detrended_surface * -1
    #Hard coded because we are looking for where 40% of the values lie
    Nl = 30 
    Nt = 70
    BAC = get_abbott_curve(detrended_surface,imageHeight, imageWidth, Nl, Nt)
    fit_x = BAC[5][BAC[3]:BAC[2]]
    fit_y = BAC[4][BAC[3]:BAC[2]]
    fit_y = np.transpose(fit_y)

    on  = np.ones(np.shape(fit_x))
    X=np.array([on,fit_x]).T
    # Linear reg
    #X = [ones(length(fit_x),1) fit_x']

    #b[1] is C, b[2] is m in y =mx +c
    #b = X\fit_y'
    #b,_,_,_ = scipy.linalg.lstsq(X, fit_y)
    b,_,_,_ = scipy.linalg.lstsq(X, fit_y[np.newaxis, :].T)
    hundred_intercept = 100 * b[1] + b[0]
    spk = np.max(BAC[4]) - b[0]
    #spk = max(BAC.zl) - b(1)
    sk = b[0] - hundred_intercept
    #sk = b(1) - hundred_intercept
    svk = (max(BAC[4]) - min(BAC[4])) - (spk + sk)
    #svk = (max(BAC.zl) - min(BAC.zl)) - (spk + sk)
    ratio_spk_sk = spk[0]/sk[0]
    ratio_svk_sk = svk[0]/sk[0]
    
    return (round(sk[0],8), round(spk[0],8), round(svk[0],8), round(ratio_spk_sk,8), round(ratio_svk_sk,8))

def get_columnised(data,imageHeight,imageWidth):
	M = np.linspace(1, imageWidth, imageWidth)
	N = np.linspace(1, imageHeight, imageHeight)
	X,Y = np.meshgrid(M,N)
	Xcoeff = X.reshape(-1)
	Ycoeff = Y.reshape(-1)
	Zcoeff = data.reshape(-1)
	return Xcoeff, Ycoeff, Zcoeff,X,Y

def plane_fit(Xcoeff, Ycoeff, Zcoeff, X, Y):
    A = np.c_[Xcoeff, Ycoeff, np.ones(Xcoeff.shape[0])]
    C,_,_,_ = scipy.linalg.lstsq(A, Zcoeff)
    plane = C[0]*X + C[1]*Y + C[2]

    return plane, C

def create_plane(C,X,Y):
    return C[0]*X + C[1]*Y + C[2]

def get_kurtosis(surface):
    return kurtosis(surface, axis=None)

def get_skew(surface):
    return skew(surface, axis=None)

def extract_surface_stats(surface_im):
    # Error check size of ROI
    surface_im_np = np.array(surface_im)
    # image dimensions
    im_height, im_width = surface_im_np.shape
    
    # New Rotate using points courtesy of patrice
    # Naive rotate.
    Xcoeff, Ycoeff, Zcoeff, X_full_im, Y_full_im = get_columnised(surface_im_np,im_height,im_width)
    im_og = np.c_[Xcoeff, Ycoeff, Zcoeff]
    
    # Take only non 0 points
    Xcoeff = Xcoeff[Zcoeff>0]
    Ycoeff = Ycoeff[Zcoeff>0]
    Zcoeff = Zcoeff[Zcoeff>0]

    # points
    points_3D = np.c_[Xcoeff, Ycoeff, Zcoeff]
    target_plane = [0,0,1]
    
    surface_reoriented, im_og_rot, angle_change = reorient_surface(points_3D, im_og, target_plane)
    #new_image =np.reshape(surface_reoriented.T[:,2],np.shape(surface_im_np))

    #Test remove 0 from rotated OG here
    vec_surface_reoriented = surface_reoriented.T[:,2]

    # Detrend
    #fitted_plane, plane_coeff = plane_fit(Xcoeff, Ycoeff, surface_reoriented.T[:,2], X_full_im, Y_full_im) #Original 13th April
    fitted_plane, plane_coeff = plane_fit(Xcoeff, Ycoeff, vec_surface_reoriented, X_full_im, Y_full_im) # TEST 14th April
    #rotated_im_og = np.reshape(im_og_rot.T[:,2],np.shape(surface_im_np)) # POTENTIALLY INCORRECT WRAPPING

    
    #Test remove 0 from rotated OG here #####
    vec_im_og_rot = im_og_rot.T[:,2]
    vec_im_og_rot[im_og[:,2] ==0 ] = fitted_plane.reshape(-1)[im_og[:,2] ==0 ]
    rotated_im_og = np.reshape(vec_im_og_rot,np.shape(fitted_plane))
    #########################################

    #detrended_image = surface_im_np - fitted_plane   
    #detrended_image = new_image - fitted_plane 
    #detrended_image = surface_im_np - fitted_plane
    detrended_image = rotated_im_og - fitted_plane # OG

    ## NOTE: 15th April 2022
    ##       Inversed the detrended data so negatives (<0) are holes and positives (>0) are mounds 
    ##       Originally >0 were depressions/holes, <0 were crests/mounds
    detrended_image = detrended_image * - 1

    # Non-zero vector
    detrended_non_zero = detrended_image[surface_im_np != 0]
    non_zero_count = detrended_non_zero.shape[0]
    zero_count = detrended_image[surface_im_np == 0].shape[0]
    
    # Plot mean
    plot_mean = round(np.mean(detrended_image),20)

    # Positive vals
    pos_vals_mean = round(np.mean(detrended_non_zero[detrended_non_zero > 0]),8)
    pos_vals_std = round(np.std(detrended_non_zero[detrended_non_zero > 0]),8)

    # Negative vals
    neg_vals_mean = round(abs(np.mean(detrended_non_zero[detrended_non_zero < 0])),8)
    neg_vals_std = round(np.std(detrended_non_zero[detrended_non_zero < 0]),8)

    #Arithmatical Mean Deviation
    amd = round((np.sum(abs(detrended_non_zero)) / non_zero_count),8)
    rms = round((np.sum(detrended_non_zero**2)/ non_zero_count)**0.5,8)

    # Calculate roughness statistics
    
    # Scipy skew, kurtosis
    skew = round(get_skew(detrended_non_zero),8)
    kurtosis = round(get_kurtosis(detrended_non_zero),8)
    # Calculate Abbott stats Spk Svk Sk
    sk, spk, svk,ratio_spk_sk, ratio_svk_sk = get_abbott_stats(detrended_non_zero, im_height, im_width)

    # Surface stats
    stats = (str(plot_mean), str(pos_vals_mean), str(pos_vals_std), str(neg_vals_mean), str(neg_vals_std), str(amd), str(rms), str(skew), str(kurtosis), str(sk), str(spk), str(svk),
            str(ratio_spk_sk), str(ratio_svk_sk), str(angle_change[0]), str(angle_change[1]), str(angle_change[2]))
    stats_num = np.array([plot_mean, pos_vals_mean, pos_vals_std, neg_vals_mean, neg_vals_std, amd, rms, skew, kurtosis, sk, spk, svk,
            ratio_spk_sk, ratio_svk_sk, angle_change[0], angle_change[1], angle_change[2]])
    # Return detrended image scale uint8
    min_val_detrended = np.min(np.min(detrended_image))
    max_val_detrended = np.max(np.max(detrended_image))
    normalised_detrended = 255 + ((detrended_image - min_val_detrended) * (255-0))/(max_val_detrended - min_val_detrended)
    return stats,stats_num, normalised_detrended, fitted_plane, plane_coeff, points_3D,detrended_image,detrended_non_zero