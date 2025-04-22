import numpy as np
import cv2
from skimage.exposure import equalize_adapthist, match_histograms
import random
import torch
from tqdm import tqdm
import os
import sys
sys.path.append('RAFT-Stereo-main')
from RaftStereo import RAFTStereo
from RaftStereo import InputPadder
from RaftStereo import GetRaftArgs
from PIL import Image
from matplotlib import pyplot as plt
import matplotlib
import open3d as o3d
from ExtractSurfaceStats import extract_surface_stats as ExtractSurfaceFeatures
from matplotlib.colors import ListedColormap
import json
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
from skimage.feature import peak_local_max


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
cv2.setRNGSeed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)             
torch.cuda.manual_seed_all(SEED)         
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class DepthMono():
    def __init__(self,
                 RaftModelPath = os.path.join("raft_models", "raftstereo-eth3d.pth"),
                 OutPutSize = (640,480),
                 PreProcessFlag = True,
                 ROI=None,
                 BorderOffset=0,
                 ScaleDisps = 20,
                 SaveFlag=True,
                 SavePath = "sediment_results", 
                 ShowScale=100, 
                 Debug=True):
        self.RaftModelPath = RaftModelPath
        self.OutPutSize = OutPutSize
        self.PreProcessFlag = PreProcessFlag
        self.DistCoef = np.asarray([[-0.42704954,  0.2422638 , -0.00071215,  0.0004687 , -0.0872903 ]])
        self.CameraMatrix = np.asarray([[2.82029177e+03, 0.00000000e+00, 2.05793575e+03],[0.00000000e+00, 2.82397145e+03, 1.52515669e+03],[0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
        self.ROI = ROI
        self.Debug = Debug
        self.ShowScale = ShowScale
        self.SaveFlag = SaveFlag
        self.SavePath = SavePath
        self.Save_Dict = {}
        self.pts1 = None
        self.pts2 = None
        self.OldH1 = None
        self.OldH2 = None
        self.H1 = None  
        self.H2 = None
        self.F = None
        self.RoiLeft = None
        self.RoiRight = None
        self.Device = 'cuda'
        self.BorderOffset = BorderOffset
        self.ScaleDisps =ScaleDisps
    def Reset(self):
        self.pts1 = None
        self.pts2 = None
        self.OldH1 = None
        self.OldH2 = None
        self.H1 = None  
        self.H2 = None
        self.F = None
        self.RoiLeft = None
        self.RoiRight = None
        self.ROI = None

    def Show(self,img):
        if self.ShowScale !=None:
            w = int(img.shape[1] * self.ShowScale / 100)
            h = int(img.shape[0] * self.ShowScale / 100)
            img = cv2.resize(img, (w, h), interpolation = cv2.INTER_AREA)
        cv2.imshow('image', img)
        k = cv2.waitKey(-1) & 0xFF
        if k == ord('s'):
            cv2.imwrite("image.png", img)
            cv2.destroyAllWindows()     
        if k == ord('q'):
            cv2.destroyAllWindows()  
        cv2.destroyAllWindows()

    def UndistortImage(self,img):
        img = img[83:3074, :] # this is specifically related to the borders that were removed during calibration
        undist = cv2.undistort(img, self.CameraMatrix, self.DistCoef, None, self.CameraMatrix)
        return undist
    
    def GetROI(self,image):
        IMAGE = image.copy()
        def draw_rectangle(event, x, y, flags, params):
            global rectangle_points, img, img_orig
            # If left mouse button is pressed down, start the crop
            if event == cv2.EVENT_LBUTTONDOWN:
                rectangle_points = [(x, y)]
            # If left mouse button is released, finish the crop
            elif event == cv2.EVENT_LBUTTONUP:
                rectangle_points.append((x, y))
                cv2.rectangle(img, rectangle_points[0], rectangle_points[1], (0, 255, 0), 2)
                cv2.imshow("image", img)
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
        if self.ROI==None:
            if len(rectangle_points)!=0:
                roi =  [(int(rectangle_points[0][0]*100/30),int(rectangle_points[0][1]*100/30)),(int(rectangle_points[1][0]*100/30),int(rectangle_points[1][1]*100/30))]
                self.ROI = roi
    
    def PreProcessImage (self, img):
        if self.PreProcessFlag:
            # reference = cv2.imread("shell_1.png")
            image = img.astype(np.float32)
            mean, std = cv2.meanStdDev(image)
            normalized_image = (image - mean.reshape(1,3)) / std.reshape(1,3)
            normalized_image = cv2.normalize(normalized_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            equalized_img = equalize_adapthist(normalized_image)
            equalized_img = cv2.normalize(equalized_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            # matched = match_histograms(normalized_image, reference, channel_axis=-1)
            img = equalized_img.copy()
            return img
        return img
    
    def ResizeToOutPutSize (self, img):
        return cv2.resize(img,self.OutPutSize)
    
    def ProcessImagePairs (self,frame1, frame2):
        if self.Debug:
            self.ShowScale//=2
            self.Show(np.hstack((frame1,frame2)))
            self.ShowScale*=2
        frame1 = self.UndistortImage(frame1)
        frame2 = self.UndistortImage(frame2)
        if self.Debug:
            self.ShowScale//=2
            self.Show(np.hstack((frame1,frame2)))
            self.ShowScale*=2
        if self.SaveFlag:
            self.Save_Dict.update({"undist_left":frame1})
            self.Save_Dict.update({"undist_right":frame2})
        if self.ROI==None:
            self.GetROI(frame1)
        roi = self.ROI
        frame1 = frame1[roi[0][1]:roi[1][1], roi[0][0]:roi[1][0]]
        frame2 = frame2[roi[0][1]:roi[1][1], roi[0][0]:roi[1][0]]
        if self.SaveFlag:
            self.Save_Dict.update({"roi_left":frame1})
            self.Save_Dict.update({"roi_right":frame2})
        frame1 = self.ResizeToOutPutSize(frame1)
        frame2 = self.ResizeToOutPutSize(frame2)
        self.RoiLeft = frame1.copy()
        self.RoiRight = frame2.copy()
        self.ShowScale+=25
        if self.Debug:
            self.Show(np.hstack((frame1,frame2)))
        frame1 = self.PreProcessImage(frame1)
        frame2 = self.PreProcessImage(frame2)
        if self.Debug:
            self.Show(np.hstack((frame1,frame2)))
        if self.SaveFlag:
            self.Save_Dict.update({"proccessed_left":frame1})
            self.Save_Dict.update({"proccessed_right":frame2})
        return frame1,frame2

    def BGR2GRAY(self,img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def GetMatches (self, img1, img2, mode="sift", ratio_threshold=0.6):
        if mode == "orb":
            orb = cv2.ORB_create()
            kp1, des1 = orb.detectAndCompute(img1, mask=None)
            kp2, des2 = orb.detectAndCompute(img2, mask=None)
            FLANN_INDEX_LSH = 6
            index_params = dict(
                algorithm=FLANN_INDEX_LSH,
                table_number=6,  # 12
                key_size=12,  # 20
                multi_probe_level=1)  
            search_params = dict(checks=50)  # or pass empty dictionary
        if mode == "sift":
            sift = cv2.SIFT_create()
            kp1, des1 = sift.detectAndCompute(img1, mask=None)
            kp2, des2 = sift.detectAndCompute(img2, mask=None)
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)   # or pass empty dictionary
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)  
        
        matchesMask = [[0, 0] for i in range(len(matches))]
        good_matches = []
        pts1 = []
        pts2 = []
        for i, (m, n) in enumerate(matches):
            if m.distance < ratio_threshold * n.distance:
                matchesMask[i] = [1,0]
                good_matches.append(m)
                pts2.append(kp2[m.trainIdx].pt)
                pts1.append(kp1[m.queryIdx].pt)
        
       
        draw_params = dict(matchColor=(0, 255, 0),
                           singlePointColor=(255, 0, 0),
                           matchesMask=matchesMask,
                           flags=cv2.DrawMatchesFlags_DEFAULT)
        img = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches, None, **draw_params)
        
        if self.Debug:
            self.Show(img)
        if self.SaveFlag:
            self.Save_Dict.update({"out_matching":img})
        
        pts1 = np.int32(pts1)
        pts2 = np.int32(pts2) 
        return pts1, pts2
    
    def Warp(self, imgL, imgR, H1, H2):
        def get_transformed_bbox(img, H):
            h, w = img.shape[:2]
            corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32).reshape(-1, 1, 2)
            warped = cv2.perspectiveTransform(corners, H)
            return warped
        warped_corners1 = get_transformed_bbox(imgL, H1)
        warped_corners2 = get_transformed_bbox(imgR, H2)
        all_corners = np.concatenate((warped_corners1, warped_corners2), axis=0)
        [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
        [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel() + 0.5)
        size = (xmax - xmin, ymax - ymin)
        offset = np.array([[1, 0, -xmin],[0, 1, -ymin],[0, 0, 1]])
        H1_adj = offset @ H1
        H2_adj = offset @ H2

        # disparities could be very small less than 1 pixels, this make sure we have large enough disparity for the stereo matching model. 
        # Later this offset will be used to get actuall disp values
        dispoffset = int(abs(np.mean(warped_corners1[:,0]-warped_corners1[:,0])))
        self.DispOffset = dispoffset
        translation_matrix = np.array([[1, 0, -dispoffset],[0, 1, -ymin],[0, 0, 1]])
        H2_adj = translation_matrix @ H2

        warped_1 = cv2.warpPerspective(imgL, H1_adj, size)
        warped_2 = cv2.warpPerspective(imgR, H2_adj, size)
        self.H1_inverse = np.linalg.inv(H1_adj)
        self.H1 = H1_adj
        self.H2 = H2_adj
        return warped_1, warped_2

    def DrawEpiLines(self,img1,img2,F,kps1,kps2):
        visualized1 = img1.copy()
        visualized2 = img2.copy()
        lines1 = cv2.computeCorrespondEpilines(kps2, 2, F).reshape(-1, 3)
        lines2 = cv2.computeCorrespondEpilines(kps1, 1, F).reshape(-1, 3)
        width = img1.shape[1]
        def draw_point_line(img, point, line, color):
            x0, y0 = map(int, [0, -line[2] / line[1]])
            x1, y1 = map(int, [width, -(line[2] + line[0] * width) / line[1]])
            cv2.line(img, (x0, y0), (x1, y1), color, 1)
            cv2.circle(img, tuple(map(int, point)), 5, color, -1)
        for (kp1, kp2, line1, line2) in zip(kps1, kps2, lines1, lines2):
            color = tuple(np.random.randint(0, 255, 3).tolist())
            draw_point_line(visualized1, kp1, line1, color)
            draw_point_line(visualized2, kp2, line2, color)
        return visualized1, visualized2


    def GetRectifiedImages(self,img1,img2):
        if self.SaveFlag:
            self.Save_Dict.update({"org_left":img1})
            self.Save_Dict.update({"org_right":img2})
        img1,img2 = self.ProcessImagePairs(img1,img2)
        pts1, pts2 = self.GetMatches(self.BGR2GRAY(img1),self.BGR2GRAY(img2))
        num_keypoints = len(pts1)
        flag = cv2.FM_7POINT if num_keypoints == 7 else cv2.FM_8POINT
        F, mask = cv2.findFundamentalMat(pts1.copy(), pts2.copy(), flag)
        # F, mask = cv2.findFundamentalMat(pts1.copy(), pts2.copy(), cv2.FM_RANSAC, 1.0, 0.99)
        pts1 = pts1[mask.ravel() == 1]
        pts2 = pts2[mask.ravel() == 1]
        self.pts1 = pts1
        self.pts2 = pts2
        self.F= F

        _, H1, H2 = cv2.stereoRectifyUncalibrated(pts1.copy(), pts2.copy(), F, self.OutPutSize)
        H1 /= H1[2, 2]
        H2 /= H2[2, 2]
        self.OldH1 = H1
        self.OldH2 = H2

        img1_rectified, img2_rectified= self.Warp(img1,img2, H1,H2)
        left_not_rectified, right_not_rectified = self.DrawEpiLines(img1, img2, F, pts1, pts2)
        not_rectified_pairs = cv2.hconcat([left_not_rectified, right_not_rectified])
        left_rectified, right_rectified=  self.Warp(left_not_rectified, right_not_rectified, H1,H2)
        rectified_pairs = cv2.hconcat([left_rectified, right_rectified])

        vertical_disparities = [abs(pts1[index][1] - pts2[index][1]) for index in range(len(pts1))]
        before_error = np.mean(vertical_disparities)    
        rectified_kps1 = cv2.perspectiveTransform(np.array([pts1.astype(np.float32)]), H1)
        rectified_kps2 = cv2.perspectiveTransform(np.array([pts2.astype(np.float32)]), H2)
        vertical_disparities = [abs(rectified_kps1[0][index][1] - rectified_kps2[0][index][1]) for index in range(len(pts1))]
        new_error = np.mean(vertical_disparities) 

        if self.Debug:
            self.Show(not_rectified_pairs)
            self.Show(rectified_pairs)
            print("Rectification Error before: ",before_error)
            print("Rectification Error now: ",new_error)

        if self.SaveFlag:
            self.Save_Dict.update({"img1_rectified":img1_rectified})
            self.Save_Dict.update({"img2_rectified":img2_rectified})
            self.Save_Dict.update({"left_not_rectified":left_not_rectified})
            self.Save_Dict.update({"right_not_rectified":right_not_rectified})
            self.Save_Dict.update({"left_rectified":left_rectified})
            self.Save_Dict.update({"right_rectified":right_rectified})
            self.Save_Dict.update({"rec_error_before":before_error})
            self.Save_Dict.update({"rec_error_after":new_error})
        return img1_rectified,img2_rectified

    def InvWarp(self,img):
        H1_inverse = self.H1_inverse
        unwarpedIMG = cv2.warpPerspective(img, H1_inverse, self.OutPutSize) 
        return unwarpedIMG
    
    def GetRaftModel (self):
        args = GetRaftArgs(restore_ckpt=self.RaftModelPath,valid_iters=50,n_downsample=1)
        model = torch.nn.DataParallel(RAFTStereo(args), device_ids=[0])
        model = model.module
        model.to(self.Device)
        model.eval()
        return model, args

    def GetDispMap(self, left, right):
        left_rgb = cv2.cvtColor(left,cv2.COLOR_BGR2RGB)
        right_rgb = cv2.cvtColor(right,cv2.COLOR_BGR2RGB)
        left_tensor = torch.from_numpy(left_rgb).permute(2, 0, 1).float()
        left_tensor = left_tensor[None].to(self.Device)
        right_tensor = torch.from_numpy(right_rgb).permute(2, 0, 1).float()
        right_tensor = right_tensor[None].to(self.Device)
        model, args = self.GetRaftModel()
        padder = InputPadder(left.shape, divis_by=32)
        image1, image2 = padder.pad(left_tensor, right_tensor)
        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=True):
                _, flow_up = model(image1, image2, iters=args.valid_iters, test_mode=True)
        flow_up = padder.unpad(flow_up).squeeze()
        disp_org = flow_up.detach().cpu().numpy().squeeze()
        disp_org = self.InvWarp(disp_org)
        disp_org-=self.DispOffset 
        disp = (disp_org-disp_org.min())/((disp_org.max()-disp_org.min())+0.0000001)
        disp = cv2.bilateralFilter(disp, d=9, sigmaColor=75, sigmaSpace=75)

        disp = disp[self.BorderOffset:disp.shape[0]-self.BorderOffset, self.BorderOffset:disp.shape[1]-self.BorderOffset]
        disp_gray = cv2.cvtColor((disp*255).astype(np.uint8),cv2.COLOR_BGR2RGB) 
        disp_color = cv2.applyColorMap(disp_gray, cv2.COLORMAP_INFERNO)
        if self.SaveFlag:
            self.Save_Dict["disp_color"]=disp_color
            self.Save_Dict["disp_gray"]=disp_gray
            self.Save_Dict["disp_org"]=disp_org
            self.Save_Dict["disp_normalized"]=disp
        
        non_processed_rectified_left, _= self.Warp(self.RoiLeft,self.RoiRight, self.H1,self.H2)
        left = self.InvWarp(non_processed_rectified_left)
        left = left[self.BorderOffset:left.shape[0]-self.BorderOffset, self.BorderOffset:left.shape[1]-self.BorderOffset]
        if self.SaveFlag:
            self.Save_Dict["unwarped_left"] = left
        if self.Debug:
            self.Show(np.hstack((left, disp_gray)))
        return left, disp

 
    def Get3DVis (self,disp, image, texture = False, get_mesh =True,getwire=False, show=True):
        scale_disps = self.ScaleDisps
        width, height = image.shape[1], image.shape[0]
        colors = image.reshape(-1, 3)
        X, Y = np.meshgrid(np.arange(width), np.arange(height))
        X = X.flatten()
        Y = Y.flatten()
        Z = disp.flatten()*scale_disps
        Z = disp.flatten()*scale_disps
        points = np.vstack((X, Y, Z)).T
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.rotate(np.array([[1, 0, 0],[0, np.cos(np.pi), -np.sin(np.pi)],[0, np.sin(np.pi), np.cos(np.pi)]]), center=(0, 0, 0))
        if texture:
            pcd.colors = o3d.utility.Vector3dVector(colors/255.0)
        if not get_mesh:
            if show:
                o3d.visualization.draw_geometries([pcd],width=1920*2, height=1080*2,window_name="3d surface")
                return pcd
            else:
                return pcd
        if get_mesh:
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
            distances = pcd.compute_nearest_neighbor_distance()
            avg_dist = np.mean(distances)
            radius = 3 * avg_dist
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd,o3d.utility.DoubleVector([radius, radius * 2]))
            if getwire== True:
                mesh = self.GetMeshWires(mesh)
            if show:
                o3d.visualization.draw_geometries([mesh],width=1920*2, height=1080*2,window_name="3d surface")
                return mesh
            else: 
                return mesh
    
    def GetMeshWires (self, mesh, show=False):
        triangles = np.asarray(mesh.triangles)
        vertices = np.asarray(mesh.vertices) 
        edges = set()
        for triangle in triangles:
            for i in range(3):
                edge = tuple(sorted((triangle[i], triangle[(i + 1) % 3])))
                edges.add(edge)
        edges = np.array(list(edges))
        line_set = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(vertices),lines=o3d.utility.Vector2iVector(edges))
        if show:
            o3d.visualization.draw_geometries([line_set],width=1920*2, height=1080*2,window_name="3d surface")
            return line_set
        else:
            return line_set
    

    def ShowFittedPlaneToMesh (self,depth,img, heat_map = True, show_plane = True):
        ##########################################################
        def fit_plane_and_heat_points(points):
            plane_model, inliers = points.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)
            [a, b, c, d] = plane_model
            distances = np.array([a * point[0] + b * point[1] + c * point[2] + d for point in points.points])
            max_dist = np.max(np.abs(distances))
            normalized_distances = distances / max_dist
            # Map distances to colors using a colormap
            colormap = matplotlib.colormaps['jet']
            colors = colormap((normalized_distances + 1) / 2)[:, :3]  # Normalize to [0, 1] and get RGB
            points.colors = o3d.utility.Vector3dVector(colors)
            return points,plane_model
        def fit_plane_and_color_points(points):
            plane_model, inliers = points.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)
            [a, b, c, d] = plane_model
            colors = []
            for point in points.points:
                if a * point[0] + b * point[1] + c * point[2] + d > 0:
                    colors.append([0, 0, 1])  # Blue for points above the plane
                else:
                    colors.append([1, 0, 0])  # Red for points below the plane
            points.colors = o3d.utility.Vector3dVector(colors)
            return points,plane_model
        def create_grid_plane_mesh(plane_model, center, size_x, size_y, grid_size=50):
            [a, b, c, d] = plane_model
            # Generate a grid of points on the plane
            x_range = np.linspace(center[0] - size_x / 2, center[0] + size_x / 2, grid_size)
            y_range = np.linspace(center[1] - size_y / 2, center[1] + size_y / 2, grid_size)
            xv, yv = np.meshgrid(x_range, y_range)
            zv = (-d - a * xv - b * yv) / c
            vertices = np.vstack((xv.flatten(), yv.flatten(), zv.flatten())).T
            # Create a list of triangles for the grid
            triangles = []
            for i in range(grid_size - 1):
                for j in range(grid_size - 1):
                    idx1 = i * grid_size + j
                    idx2 = idx1 + 1
                    idx3 = idx1 + grid_size
                    idx4 = idx3 + 1
                    triangles.append([idx1, idx2, idx3])
                    triangles.append([idx2, idx4, idx3])
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(vertices)
            mesh.triangles = o3d.utility.Vector3iVector(triangles)
            mesh.paint_uniform_color([0.5, 0.5, 0.5])  # Grey color for the plane
            return mesh
        ##########################################################
        mesh = self.Get3DVis(depth,img,texture=False,get_mesh =True,show=False)
        mesh_wires= self.GetMeshWires(mesh, show=False)
        points = o3d.geometry.PointCloud()
        points.points = mesh_wires.points
        if not heat_map:
            points,plane_model = fit_plane_and_color_points(points)
        else:
            points,plane_model = fit_plane_and_heat_points(points)
        points.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        distances = points.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances)
        radius = 3 * avg_dist
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(points,o3d.utility.DoubleVector([radius, radius * 2]))
        bounding_box = points.get_axis_aligned_bounding_box()
        size_x = bounding_box.get_extent()[0]+50
        size_y = bounding_box.get_extent()[1]+50
        # Create a mesh for the plane
        center = points.get_center()
        plane_mesh = create_grid_plane_mesh(plane_model, center, size_x, size_y)
        if show_plane:
            plane = self.GetMeshWires (plane_mesh, show=False)
            o3d.visualization.draw_geometries([mesh,plane],width=1920*2, height=1080*2,window_name="3d surface")
            return mesh, plane
        else:
            o3d.visualization.draw_geometries([mesh],width=1920*2, height=1080*2,window_name="3d surface")
            plane = None
            return mesh, plane

    def GetPeaksPits (self,depth, show3d = False):
        def compute_topological_features(depth_map):
            peaks = peak_local_max(depth_map,min_distance= 1)
            pits = peak_local_max(-depth_map,min_distance= 1)
            return peaks, pits
        depth_rgb = np.ones((depth.shape[0],depth.shape[1],3)).astype(np.uint8)
        d = (((depth-depth.min())/(depth.max()-depth.min()))*255).astype(np.uint8)
        peaks, pits= compute_topological_features(d)
        depth_rgb[:,:,0] =d
        depth_rgb[:,:,1] =d
        depth_rgb[:,:,2] =d
        threshold = np.percentile(depth, 95)
        peaks = [p for p in peaks if depth[tuple(p)] > threshold]
        threshold_pit = np.percentile(depth, 5)
        pits = [p for p in pits if depth[tuple(p)] < threshold_pit]
        for point in peaks:
            depth_rgb = cv2.circle(depth_rgb,(point[1],(point[0])),3,(255,0,0),-1)
        for point in pits:
            depth_rgb = cv2.circle(depth_rgb,(point[1],(point[0])),3,(0,0,255),-1)
        if show3d:
            self.Get3DVis(depth, depth_rgb, texture = True, get_mesh =True, show=True)
        return len(pits), len(peaks) # they should be swaped since the depth is inversely proportional to depth

    def GetSurfaceStats(self, depth):
        _,stats_num, _, _, _, _,detrended_depth,detrended_non_zero = ExtractSurfaceFeatures(depth)
        detrended_depth_norm = (((detrended_depth-detrended_depth.min())/((detrended_depth.max()-detrended_depth.min())+0.0000001))*255).astype(np.uint8)
        if self.SaveFlag:
            self.Save_Dict["stats_num"]= stats_num
            self.Save_Dict["detrended_depth"]= detrended_depth_norm
        
        # stats_num includes the following in order
        # plot_mean, pos_vals_mean, pos_vals_std, neg_vals_mean, neg_vals_std, amd, rms, skew, kurtosis, sk, spk, svk,
        # ratio_spk_sk, ratio_svk_sk, angle_change[0], angle_change[1], angle_change[2]]

        Mp = stats_num [1] # Mean Peak
        sigma_p = stats_num [2] #Peak standard deviation
        Mt = stats_num [3] # Mean trough
        sigma_t = stats_num [4] # trough standard deviation
        Sa = stats_num [5] # Arithmetical mean roughness
        Sq = stats_num [6] # Root mean square Deviation
        skew = stats_num [7] # skew
        kurtosis = stats_num [8] # kurtosis  
        sk = stats_num [9] 
        spk = stats_num [10]
        svk = stats_num [11]
        ratio_spk_sk = stats_num [12]
        ratio_svk_sk = stats_num [13]
        features = [Sa, Sq, Mp, sigma_p, Mt, sigma_t,skew,kurtosis,sk,spk,svk, ratio_spk_sk, ratio_svk_sk]

        detrended_surface = detrended_depth
        z = detrended_non_zero.flatten()
        z_sorted = np.sort(z)
        percent_above = 100 * (1.0 - np.arange(len(z_sorted)) / len(z_sorted))
        # Calculate thresholds
        z_0 = z_sorted[0]
        z_100 = z_sorted[-1]
        z_30 = np.percentile(z_sorted, 70)
        z_70 = np.percentile(z_sorted, 30)
        # Abbott metrics
        Sk = z_70 - z_30
        Spk = z_100 - z_70
        Svk = z_30 - z_0
        fontsize = 16
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), gridspec_kw={'height_ratios': [2, 1]})
        ax1.plot(z_sorted, percent_above, color='black', label="Abbott-Firestone Curve")
        ax1.axvline(z_70, color='gold', linestyle='--', label='$z_{30\\%}$ (Start of Sk)')
        ax1.axvline(z_30, color='teal', linestyle='--', label='$z_{70\\%}$ (End of Sk)')
        ax1.axvline(z_100, color='orange', linestyle=':', label='Max ($z_{0\\%}$)')
        ax1.axvline(z_0, color='purple', linestyle=':', label='Min ($z_{100\\%}$)')
        ax1.fill_betweenx(percent_above, z_70, z_100, color='gold', alpha=0.3, label='Spk Region')
        ax1.fill_betweenx(percent_above, z_30, z_70, color='lightgreen', alpha=0.3, label='Sk Region')
        ax1.fill_betweenx(percent_above, z_0, z_30, color='plum', alpha=0.3, label='Svk Region')
        ax1.set_title("Abbott-Firestone Curve with Surface Functionality Zones", fontsize=fontsize+2)
        ax1.set_xlabel("Surface Height ($Z_d$)", fontsize=fontsize)
        ax1.set_ylabel("Cumulative Area Above Height (%)", fontsize=fontsize)
        ax1.tick_params(axis='both', labelsize=fontsize-2)

        ax1.legend(fontsize=fontsize)

        ax1.grid(True)
        profile_line = detrended_surface[detrended_surface.shape[0]//2 , :]
        colors = np.full(profile_line.shape, 'gray')
        colors[profile_line > z_70] = 'gold'
        colors[(profile_line <= z_70) & (profile_line >= z_30)] = 'lightgreen'
        colors[profile_line < z_30] = 'plum'
        for i in range(len(profile_line)):
            ax2.plot([i, i], [0, profile_line[i]], color=colors[i], linewidth=2)
        ax2.axhline(z_70, color='gold', linestyle='--')
        ax2.axhline(z_30, color='teal', linestyle='--')
        ax2.axhline(z_100, color='orange', linestyle=':')
        ax2.axhline(z_0, color='purple', linestyle=':')
        ax2.set_title("Surface Cross-Section with Sk, Spk, Svk Regions", fontsize=fontsize+2)
        ax2.set_xlabel("Pixel Index", fontsize=fontsize)
        ax2.set_ylabel("Height ($Z_d$)", fontsize=fontsize)
        ax2.tick_params(axis='both', labelsize=fontsize-2)
        ax2.set_ylim(z_0-0.3, z_100+0.3)
        ax2.grid(True)
        plt.tight_layout()
        if self.SaveFlag:
            if not os.path.exists(self.SavePath):
                os.makedirs(self.SavePath)
            plt.savefig(os.path.join(self.SavePath, "abbot_curve.png")) 
            plt.close()
            for key in self.Save_Dict.keys():
                if key not in ["rec_error_before","rec_error_after", "disp_org", "disp_normalized", "stats_num"]:
                    cv2.imwrite(os.path.join(self.SavePath, key+".png"),self.Save_Dict[key])
        return features,detrended_depth_norm