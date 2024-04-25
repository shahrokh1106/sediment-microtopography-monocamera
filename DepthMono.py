
from utilities import *
class DepthMono():
    def __init__(self, OutPutSize = (640,480), PreProcessFlag = True, Debug=True, ShowScale=100, ROI=None, Save=True, SavePath = "images/"):
        self.OutPutSize = OutPutSize
        self.PreProcessFlag = PreProcessFlag
        self.DistCoef = np.asarray([[-0.42704954,  0.2422638 , -0.00071215,  0.0004687 , -0.0872903 ]])
        self.CameraMatrix = np.asarray([[2.82029177e+03, 0.00000000e+00, 2.05793575e+03],[0.00000000e+00, 2.82397145e+03, 1.52515669e+03],[0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
        self.MatchingModel,self.Device = self.GetMatchingModel()
        self.Debug = Debug
        self.ShowScale = ShowScale
        self.DispModelPath = "CREStereomaster/crestereo_eth3d.mge"
        self.model = self.GetDispModel()
        self.H1_inverse = None
        self.Save = Save
        self.SavePath = SavePath
        self.ROI = ROI
        self.Save_Dict = {}
        self.pts1 = None
        self.pts2 = None
        self.OldH1 = None
        self.OldH2 = None
        self.H1 = None
        self.H2 = None
        self.Disp = None
        self.F = None
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
    def Warp2(self,imgL,imgR,H1,H2): 
        h, w = imgL.shape[:2]
        warped_1 = cv2.warpPerspective(imgL, H1, (w, h))
        warped_2 = cv2.warpPerspective(imgR, H2, (w, h))
        H1_inverse = np.linalg.inv(H1)
        self.H1_inverse = H1_inverse
        self.H1 =  H1
        self.H2 =  H2
        return warped_1, warped_2

    
    def Warp(self,img1,img2,H1,H2):  
        h, w = img1.shape[:2]
        corners_1 = np.array([[0, 0],[0, h - 1],[w - 1, h - 1],[w - 1, 0]])
        warped_corners_1 = cv2.perspectiveTransform(np.float32([corners_1]), H1)[0]
        x_min = int(np.min(warped_corners_1[:, 0]))
        x_max = int(np.max(warped_corners_1[:, 0]))
        y_min = int(np.min(warped_corners_1[:, 1]))
        y_max = int(np.max(warped_corners_1[:, 1]))
        width = x_max - x_min
        height = y_max - y_min
        translation_matrixx = np.array([[1, 0, -x_min],[0, 1, -y_min],[0, 0, 1]])
        warped_1 = cv2.warpPerspective(img1, np.dot(translation_matrixx, H1), (width, height))
        self.H1 = np.dot(translation_matrixx, H1)
        H1_inverse = np.linalg.inv(np.dot(translation_matrixx, H1))
        self.H1_inverse = H1_inverse
        corners_2 = np.array([[0, 0],[0, h - 1],[w - 1, h - 1],[w - 1, 0]])
        warped_corners_2 = cv2.perspectiveTransform(np.float32([corners_2]), H2)[0]
        offset = int(abs(np.mean(warped_corners_1[:,0]-warped_corners_2[:,0])))
        translation_matrix = np.array([[1, 0, -offset],[0, 1, -y_min],[0, 0, 1]])
        warped_2 = cv2.warpPerspective(img2, np.dot(translation_matrix, H2), (width, height))
        self.H2 = np.dot(translation_matrix, H2)
        return warped_1, warped_2
        
    def ResizeToOutPutSize (self, img):
        return cv2.resize(img,self.OutPutSize)
    def BGR2GRAY(self,img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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
    def UndistortImage(self,img):
        img = img[83:3074, :]
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
        if self.ROI==None:
            if len(rectangle_points)!=0:
                roi =  [(int(rectangle_points[0][0]*100/30),int(rectangle_points[0][1]*100/30)),(int(rectangle_points[1][0]*100/30),int(rectangle_points[1][1]*100/30))]
                self.ROI = roi
    def GetMatchingModel(self):
        max_keypoints=500
        keypoint_threshold=0.001
        nms_radius = 4
        match_threshold = 0.1
        superglue='outdoor'
        sinkhorn_iterations = 20
        torch.set_grad_enabled(False);
        device = 'cuda' if torch.cuda.is_available()  else 'cpu'
        config = {
                'superpoint': {
                    'nms_radius': nms_radius,
                    'keypoint_threshold':keypoint_threshold,
                    'max_keypoints': max_keypoints
                },
                'superglue': {
                    'weights': superglue,
                    'sinkhorn_iterations': sinkhorn_iterations,
                    'match_threshold': match_threshold,
                }
            }
        matching = Matching(config).eval().to(device)
        
        return matching, device
        
    def get_matches (self, img1, img2, mode="sift", ratio_threshold=0.6):
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
        
       
        draw_params = dict(matchColor=(0, 0, 255),
                           singlePointColor=(255, 0, 0),
                           matchesMask=matchesMask,
                           flags=cv2.DrawMatchesFlags_DEFAULT)
        img = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches, None, **draw_params)
        
        if self.Debug:
            self.Show(img)
        if self.Save:
            self.Save_Dict.update({"out_matching":img})
        
        pts1 = np.int32(pts1)
        pts2 = np.int32(pts2) 
        return pts1, pts2

    
 
  
    def GetPointCorrespondences (self,img1,img2,max_keypoints=500,keypoint_threshold=0.001,nms_radius = 4,match_threshold = 0.1):
        input = 0
        skip = 1
        max_length=1000000
        resize=[640, 480]
        superglue='outdoor'
        sinkhorn_iterations = 20
        keys = ['keypoints', 'scores', 'descriptors']
        frame_tensor = frame2tensor(img1, self.Device)
        last_data = self.MatchingModel.superpoint({'image': frame_tensor})
        last_data = {k+'0': last_data[k] for k in keys}
        last_data['image0'] = frame_tensor
        last_frame = img1
        last_image_id = 0
        frame_tensor = frame2tensor(img2, self.Device)
        pred = self.MatchingModel({**last_data, 'image1': frame_tensor})
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
        k_thresh = self.MatchingModel.superpoint.config['keypoint_threshold']
        m_thresh = self.MatchingModel.superglue.config['match_threshold']
        small_text = [
            'Keypoint Threshold: {:.4f}'.format(k_thresh),
            'Match Threshold: {:.2f}'.format(m_thresh),
        ]
        show_keypoints='store_true'
        out = make_matching_plot_fast(
            last_frame, img2, kpts0, kpts1, mkpts0, mkpts1, color, text,
            path=None, show_keypoints=show_keypoints, small_text=small_text)
        if self.Debug:
            self.Show(out)
        if self.Save:
            self.Save_Dict.update({"out_matching":out})
        return mkpts0,mkpts1

    
    def GetRectifiedImages(self,img1,img2):
        def draw_epi_lines(img1,img2,F,kps1,kps2,seed=42):
            np.random.seed(seed)
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

        if self.Save:
            self.Save_Dict.update({"org_left":img1})
            self.Save_Dict.update({"org_right":img2})
            
        img1 = self.UndistortImage(img1)
        img2 = self.UndistortImage(img2)
        if self.Save:
            self.Save_Dict.update({"undist_left":img1})
            self.Save_Dict.update({"undist_right":img2})
        if self.ROI==None:
            self.GetROI(img1)
        roi = self.ROI
        img1 = img1[roi[0][1]:roi[1][1], roi[0][0]:roi[1][0]]
        img2 = img2[roi[0][1]:roi[1][1], roi[0][0]:roi[1][0]]
        if self.Save:
            self.Save_Dict.update({"roi_left":img1})
            self.Save_Dict.update({"roi_right":img2})

        img1 = self.ResizeToOutPutSize(img1)
        img2 = self.ResizeToOutPutSize(img2)
        self.OutPutSize = (img1.shape[1],img1.shape[0])
        if self.Debug:
            self.Show(np.hstack((img1,img2)))
        if self.PreProcessFlag:
            img1 = self.PreProcessImage(img1)
            img2 = self.PreProcessImage(img2)
            if self.Debug:
                self.Show(np.hstack((img1,img2)))
            if self.Save:
                self.Save_Dict.update({"proccessed_left":img1})
                self.Save_Dict.update({"proccessed_right":img2})

        # mkpts0,mkpts1 = self.get_matches (self.BGR2GRAY(img1), self.BGR2GRAY(img2))
        mkpts0,mkpts1 = self.GetPointCorrespondences(self.BGR2GRAY(img1), self.BGR2GRAY(img2))
        num_keypoints = len(mkpts0)
        flag = cv2.FM_7POINT if num_keypoints == 7 else cv2.FM_8POINT
        F, mask = cv2.findFundamentalMat(mkpts0.copy(), mkpts1.copy(), flag)
        pts1 = mkpts0[mask.ravel() == 1]
        pts2 = mkpts1[mask.ravel() == 1]
        self.pts1 = pts1
        self.pts2 = pts2
        self.F= F
        _, H1, H2 = cv2.stereoRectifyUncalibrated(pts1.copy(), pts2.copy(), F, self.OutPutSize)
        H1 /= H1[2, 2]
        H2 /= H2[2, 2]
        self.OldH1 = H1
        self.OldH2 = H2
        img1_rectified, img2_rectified= self.Warp(img1,img2, H1,H2)

        left_not_rectified, right_not_rectified = draw_epi_lines(img1, img2, F, pts1, pts2)
        not_rectified_pairs = cv2.hconcat([left_not_rectified, right_not_rectified])
        left_rectified, right_rectified=  self.Warp(left_not_rectified, right_not_rectified, H1,H2)
        rectified_pairs = cv2.hconcat([left_rectified, right_rectified])
        if self.Debug:
            self.Show(not_rectified_pairs)
            self.Show(rectified_pairs)
            vertical_disparities = [abs(pts1[index][1] - pts2[index][1]) for index in range(len(pts1))]
            before_error = np.mean(vertical_disparities)    
            rectified_kps1 = cv2.perspectiveTransform(np.array([pts1.astype(np.float32)]), H1)
            rectified_kps2 = cv2.perspectiveTransform(np.array([pts2.astype(np.float32)]), H2)
            vertical_disparities = [abs(rectified_kps1[0][index][1] - rectified_kps2[0][index][1]) for index in range(len(pts1))]
            new_error = np.mean(vertical_disparities) 
            print("Rectification Error before: ",before_error)
            print("Rectification Error now: ",new_error)
        if self.Save:
            self.Save_Dict.update({"img1_rectified":img1_rectified})
            self.Save_Dict.update({"img2_rectified":img2_rectified})
            self.Save_Dict.update({"left_not_rectified":left_not_rectified})
            self.Save_Dict.update({"right_not_rectified":right_not_rectified})
            self.Save_Dict.update({"left_rectified":left_rectified})
            self.Save_Dict.update({"right_rectified":right_rectified})
        return img1_rectified,img2_rectified
        
    def invWarp(self,warped_1,disp):
        H1_inverse = self.H1_inverse
        unwarpedIMG = cv2.warpPerspective(warped_1, H1_inverse, self.OutPutSize) 
        unwarpedDISP = cv2.warpPerspective(disp, H1_inverse, self.OutPutSize) 
        offset = 20
        unwarpedIMG = unwarpedIMG[offset:unwarpedIMG.shape[0]-offset, offset:unwarpedIMG.shape[1]-offset]
        unwarpedDISP = unwarpedDISP[offset:unwarpedDISP.shape[0]-offset, offset:unwarpedDISP.shape[1]-offset]

        if self.Debug==True:
            unwarpedDISP_VIS = (unwarpedDISP - unwarpedDISP.min()) / (unwarpedDISP.max() - unwarpedDISP.min()) * 255.0
            unwarpedDISP_VIS = unwarpedDISP_VIS.astype("uint8")
            unwarpedDISP_VIS = cv2.applyColorMap(unwarpedDISP_VIS, cv2.COLORMAP_INFERNO)
            self.Show(np.hstack((unwarpedIMG,unwarpedDISP_VIS)))
        return unwarpedIMG,unwarpedDISP
    def GetDispModel (self):
        pretrained_dict = mge.load(self.DispModelPath)
        model = Model(max_disp=256, mixed_precision=False, test_mode=True)
        model.load_state_dict(pretrained_dict["state_dict"], strict=True)
        model.eval()
        return model

    def ComputeDisp(self, left,right,n_iter=30):
        def inference(left, right, model, n_iter=30):
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
        model = self.model
        in_h, in_w = left.shape[:2]
        eval_h, eval_w = [int(e) for e in (1024,1536)]
        left_img = cv2.resize(left, (eval_w, eval_h), interpolation=cv2.INTER_LINEAR)
        right_img = cv2.resize(right, (eval_w, eval_h), interpolation=cv2.INTER_LINEAR)
        pred = inference(left_img, right_img, model, n_iter=n_iter)
        t = float(in_w) / float(eval_w)
        disp = cv2.resize(pred, (in_w, in_h), interpolation=cv2.INTER_LINEAR) * t
        disp_vis = (disp - disp.min()) / (disp.max() - disp.min()) * 255.0
        disp_vis = disp_vis.astype("uint8")
        disp_vis = cv2.applyColorMap(disp_vis, cv2.COLORMAP_INFERNO)
        if self.Debug:
            self.Show(disp_vis)
        self.Disp = disp
        return disp

    def Get3d(self, img, disp,sample_index):
        w,h= img.shape[1],img.shape[0]
        cx, cy = w//2, h//2
        f = 2.32029177e+03
        b=150.3212366038011
        Q = np.asarray([[1,0,0,cx],[0,1,0,cy],[0,0,0,(f)],[0,0,1/b,0]])
        save_point_cloud(img, disp, Q ,name=self.SavePath+"/"+str(sample_index)+"/3d", mask=np.zeros(disp.shape))
        
    def GetDisp(self, img1, img2):
        rectified_1, rectified2 = self.GetRectifiedImages(img1,img2)
        disp = self.ComputeDisp(rectified_1,rectified2)
        img,disp = self.invWarp(rectified_1,disp)
        if self.Save:
            self.Save_Dict.update({"unwarped_left":img})
            disp_vis = (disp - disp.min()) / (disp.max() - disp.min()) * 255.0
            disp_vis = disp_vis.astype("uint8")
            disp_vis = cv2.applyColorMap(disp_vis, cv2.COLORMAP_INFERNO)
            self.Save_Dict.update({"unwarped_disp_vis":disp_vis})
            self.Save_Dict.update({"disp":disp})
    def Run(self, img1,img2):
        self.GetDisp(img1,img2)

    def GetDispError(self):
        pts1 = np.float32(self.pts1)
        pts2 = np.float32(self.pts2)
        H1 = self.H1
        H2 = self.H2
        imgg1 = self.Save_Dict["img1_rectified"].copy()
        w,h = imgg1.shape[1],imgg1.shape[0]
        disp = DispPipeline.Disp
        ones = np.ones((pts1.shape[0],1))
        rectified_pts1 = cv2.perspectiveTransform(np.expand_dims(pts1,1), H1)[:,0,:]
        rectified_pts2 = cv2.perspectiveTransform(np.expand_dims(pts2,1), H2)[:,0,:]
        disp_values = np.abs(rectified_pts2[:,0]-rectified_pts1[:,0])
        pixels = np.int32(rectified_pts1)
        sum = 0
        c = 0
        for index in range(len(pixels)):
            x =pixels[index][0]
            y = pixels[index][1]
            if y>=0 and x>=0 and y<h and x<w:
                sum+=abs(disp[y,x]-disp_values[index])
                c+=1
        if c!=0:
            sum = sum/c
        return sum
        

    def Save_all(self, sample_index):
        if len(self.Save_Dict)!=0:
            folder_path = self.SavePath+"/"+str(sample_index)+"/"
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            cv2.imwrite(folder_path+"out_matching.png", self.Save_Dict["out_matching"])
            cv2.imwrite(folder_path+"org_left.png", self.Save_Dict["org_left"])
            cv2.imwrite(folder_path+"org_right.png", self.Save_Dict["org_right"])
            cv2.imwrite(folder_path+"undist_left.png", self.Save_Dict["undist_left"])
            cv2.imwrite(folder_path+"undist_right.png", self.Save_Dict["undist_right"])
            cv2.imwrite(folder_path+"roi_left.png", self.Save_Dict["roi_left"])
            cv2.imwrite(folder_path+"roi_right.png", self.Save_Dict["roi_right"])
            cv2.imwrite(folder_path+"img1_rectified.png", self.Save_Dict["img1_rectified"])
            cv2.imwrite(folder_path+"img2_rectified.png", self.Save_Dict["img2_rectified"])
            cv2.imwrite(folder_path+"left_not_rectified.png", self.Save_Dict["left_not_rectified"])
            cv2.imwrite(folder_path+"right_not_rectified.png", self.Save_Dict["right_not_rectified"])
            cv2.imwrite(folder_path+"left_rectified.png", self.Save_Dict["left_rectified"])
            cv2.imwrite(folder_path+"right_rectified.png", self.Save_Dict["right_rectified"])
            cv2.imwrite(folder_path+"unwarped_left.png", self.Save_Dict["unwarped_left"])
            cv2.imwrite(folder_path+"unwarped_disp_vis.png", self.Save_Dict["unwarped_disp_vis"])
            np.save(folder_path+"disp.npy", self.Save_Dict["disp"])
            self.Get3d(self.Save_Dict["unwarped_left"],self.Save_Dict["disp"], sample_index)


def main():
    DispPipeline = DepthMono(PreProcessFlag=True,ShowScale=50, SavePath='D:/sediment_results/',Debug = False)   
    video_name = "Site 12 drift.mkv"
    if not os.path.exists(DispPipeline.SavePath+video_name):
        os.makedirs(DispPipeline.SavePath+video_name)
    DispPipeline.SavePath = DispPipeline.SavePath+video_name+"/"
    input_path = "data/trimmed/"+video_name
    cap = cv2.VideoCapture(input_path)
    ret, frame = cap.read()
    k = 500
    cv2.waitKey(k)
    index = 1
    while cap.isOpened():
        ret1, frame1 = cap.read()
        cv2.waitKey(k+5000) 
        ret2, frame2 = cap.read()
        DispPipeline.Run(frame1, frame2)
        DispPipeline.Save_all(sample_index = index)
        index+=1
        cv2.waitKey(k+2000)
    cap.release()
    cv2.destroyAllWindows() 

if __name__ == "__main__":
    main()
