from DepthMonoCore import *
import glob
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import folium
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, fftshift
from scipy.stats import linregress, entropy
from scipy.signal import windows
from scipy.optimize import curve_fit
def GetFeatures():
    DepthPipeline = DepthMono(RaftModelPath = os.path.join("raft_models", "raftstereo-eth3d.pth"),
                             OutPutSize = (640,480),
                             PreProcessFlag=True,
                             ROI= None,
                             BorderOffset = 50,
                             ScaleDisps =20,
                             SaveFlag=True,
                             SavePath = "dataset",
                             ShowScale=50,
                             Debug = False)
    sites = ["6a","7b"]
    features_flag= True
    features_dict = {}

    for site in sites:
        if not os.path.exists(os.path.join("dataset",site,"features.json")):
            features_flag = False
    if features_flag:
        for site in sites:
            features_path = os.path.join("dataset",site,"features.json")
            with open(features_path, "r") as f:
                Features = json.load(f)
            features_dict[site]=Features
        return features_dict
    
    ROIs = [[(1021, 1226), (2913, 2533)],[(1056, 1193), (2970, 2576)]]
    for index, site in enumerate(sites):
        Features = {}
        sample_folders = [f for f in glob.glob(os.path.join("dataset", site, "*")) if os.path.isdir(f)]
        DepthPipeline.ROI = ROIs[index]
        print("Getting features for Site "+site)
        for sample_folder in sample_folders:
            sample_name = os.path.basename(sample_folder)
            left_path = os.path.join(sample_folder,"left.png")
            right_path = os.path.join(sample_folder,"right.png")
            left = cv2.imread(left_path)
            right = cv2.imread(right_path)
            out_path = os.path.join(sample_folder,"sediment_results")
            DepthPipeline.SavePath = out_path
            rectified_left,rectified_right = DepthPipeline.GetRectifiedImages(left,right)
            l,d = DepthPipeline.GetDispMap(rectified_left, rectified_right)
            print(sample_name)
            mesh = DepthPipeline.Get3DVis(d,l, texture = True, get_mesh =False,getwire=False, show=True)
            # DepthPipeline.ShowFittedPlaneToMesh (d,l, heat_map = True, show_plane = True)
            features,_ = DepthPipeline.GetSurfaceStats(d)
            Features[sample_name] = list(features)
            DepthPipeline.ShowScale = 50
        with open(os.path.join("dataset",site,"features.json"), "w") as f:
            json.dump(Features, f, indent=4)
        features_dict[site]=Features
        DepthPipeline.ROI = None
    return features_dict


def BuildDataframeFromFeatures():
    features_dict = GetFeatures()
    feature_names = ["Sa", "Sq", "Mp", "sigma_p", "Mt", "sigma_t", "skew", "kurtosis","sk","spk","svk", "spk/sk", "svk/sk"]
    records = []
    for class_label, samples in features_dict.items():
        if class_label=="6a":
            class_label = "Sand"
        else:
            class_label="Shell-Hash"
        for sample_name, feature_set in samples.items():
            entry = {"class": class_label}
            entry.update(dict(zip(feature_names, feature_set)))
            records.append(entry)
    df = pd.DataFrame(records)
    return df, feature_names

def GetRadarPlot(A,B, plot_name):
    labels = ["Sa", "Sq", "Mp", "sigma_p", "Mt", "sigma_t", "skew", "kurtosis","sk", "spk", "svk", "spk/sk", "svk/sk"]
    max_vals = np.maximum(A, B)
    A_norm = A / max_vals
    B_norm = B / max_vals
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    A_plot = np.concatenate((A_norm, [A_norm[0]]))
    B_plot = np.concatenate((B_norm, [B_norm[0]]))
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(8, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, A_plot, label='Sand', linewidth=2)
    ax.plot(angles, B_plot, label='Shell-Hash', linewidth=2)
    ax.fill(angles, A_plot, alpha=0.25)
    ax.fill(angles, B_plot, alpha=0.25)
    ax.set_thetagrids([])
    label_radius = 1.3  
    for angle, label in zip(angles[:-1], labels):
        ax.text(angle,label_radius,label,size=18,horizontalalignment='center',verticalalignment='center')
    ax.set_title("Microtopographic Features (Normalized)", fontsize=18, pad=55)
    ax.legend(loc='upper left', bbox_to_anchor=(1.09, 1.0), borderaxespad=0., fontsize=18)
    plt.tight_layout()
    plt.savefig(os.path.join("dataset",plot_name+".png"))
    plt.close()

def GetCenterSquare(image, ratio = 0.5):
    H, W = image.shape[:2]
    square_size = int(min(H, W) * ratio)
    center_y, center_x = H // 2, W // 2
    half_size = square_size // 2
    y1 = max(center_y - half_size, 0)
    y2 = y1 + square_size
    x1 = max(center_x - half_size, 0)
    x2 = x1 + square_size
    return image[y1:y2, x1:x2]

def split_mesh_along_x_with_gap(mesh, gap=20.0):
    verts = np.asarray(mesh.vertices)
    tris = np.asarray(mesh.triangles)
    x_coords = verts[:, 0]
    x_mid = (x_coords.min() + x_coords.max()) / 2
    left_mask = x_coords < x_mid
    right_mask = ~left_mask
    left_indices = np.where(left_mask)[0]
    right_indices = np.where(right_mask)[0]
    def extract_half(selected_indices, shift):
        idx_map = {old_idx: new_idx for new_idx, old_idx in enumerate(selected_indices)}
        selected_verts = verts[selected_indices]
        
        # Add this if vertex colors exist
        if mesh.has_vertex_colors():
            all_colors = np.asarray(mesh.vertex_colors)
            selected_colors = all_colors[selected_indices]
        else:
            selected_colors = None

        valid_tris = [tri for tri in tris if all(v in idx_map for v in tri)]
        reindexed_tris = [[idx_map[v] for v in tri] for tri in valid_tris]

        submesh = o3d.geometry.TriangleMesh()
        submesh.vertices = o3d.utility.Vector3dVector(selected_verts)
        submesh.triangles = o3d.utility.Vector3iVector(reindexed_tris)
        
        if selected_colors is not None:
            submesh.vertex_colors = o3d.utility.Vector3dVector(selected_colors)
        
        submesh.compute_vertex_normals()
        submesh.translate((shift, 0, 0))
        return submesh
    left_mesh = extract_half(left_indices, shift=-gap / 2)
    right_mesh = extract_half(right_indices, shift=gap / 2)
    return left_mesh, right_mesh

def compute_overlap_area(DepthPipeline,img1, img2):
        img1 = DepthPipeline.PreProcessImage(img1)
        img2 = DepthPipeline.PreProcessImage(img2)
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        orb = cv2.ORB_create(5000)
        kp1, des1 = orb.detectAndCompute(gray1, None)
        kp2, des2 = orb.detectAndCompute(gray2, None)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        if len(matches) < 10:
            return 0.0  # Not enough matches
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if H is None:
            return 0.0
        h, w = img1.shape[:2]
        warped_img1 = cv2.warpPerspective(img1, H, (w, h))

        mask1 = cv2.cvtColor(warped_img1, cv2.COLOR_BGR2GRAY) > 0
        mask2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) > 0
        # Compute overlap
        intersection = np.logical_and(mask1, mask2)
        union = np.logical_or(mask1, mask2)

        overlap_percent = (np.sum(intersection) / np.sum(union)) * 100
        return overlap_percent

def DoExperimentAll(experiment,experiment_index, show3d = False):
    ratio_crop = 0.7
    DepthPipeline = DepthMono(RaftModelPath = os.path.join("raft_models", "raftstereo-eth3d.pth"),
                            OutPutSize = (640,480),
                            PreProcessFlag=True,
                            ROI= None,
                            BorderOffset = 50,
                            ScaleDisps =10,
                            SaveFlag=False,
                            SavePath = "dataset",
                            ShowScale=50,
                            Debug = True)
    ROIs = [[(1021, 1226), (2913, 2533)],[(1056, 1193), (2970, 2576)]]
    sand_left_path = os.path.join("dataset","6a",str(experiment[0]), "left.png")
    sand_right_path = os.path.join("dataset","6a" ,str(experiment[0]), "right.png")
    sand_left = cv2.imread(sand_left_path)
    sand_right = cv2.imread(sand_right_path)
    shell_left_path = os.path.join("dataset","7b" ,str(experiment[1]), "left.png")
    shell_right_path = os.path.join("dataset","7b" ,str(experiment[1]), "right.png")
    shell_left = cv2.imread(shell_left_path)
    shell_right = cv2.imread(shell_right_path)

    DepthPipeline.ROI = ROIs[0]
    rleft,rright = DepthPipeline.GetRectifiedImages(sand_left,sand_right)
    sand_l,sand_d = DepthPipeline.GetDispMap(rleft, rright)
    sand_l = GetCenterSquare(sand_l, ratio = ratio_crop)
    sand_d = GetCenterSquare(sand_d, ratio = ratio_crop)
    sand_features, sand_detrened = DepthPipeline.GetSurfaceStats(sand_d)
    sand_features = np.asarray(sand_features)
    DepthPipeline.ROI = None

    DepthPipeline.ROI = ROIs[1]
    rleft,rright = DepthPipeline.GetRectifiedImages(shell_left,shell_right)
    shell_l,shell_d = DepthPipeline.GetDispMap(rleft, rright)
    shell_l = GetCenterSquare(shell_l, ratio = ratio_crop)
    shell_d = GetCenterSquare(shell_d, ratio = ratio_crop)
    shell_features,shell_detrened = DepthPipeline.GetSurfaceStats(shell_d)
    shell_features = np.asarray(shell_features)
    DepthPipeline.ROI = None

    plot_name = "experiment_"+str(experiment_index)+"_radar_plot"
    print(plot_name, experiment)
    GetRadarPlot(sand_features.copy(),shell_features.copy(),plot_name)
    if show3d:
        shell_mesh = DepthPipeline.Get3DVis(shell_d,shell_l, texture = False, get_mesh =True,getwire=False, show=False)
        sand_mesh = DepthPipeline.Get3DVis(sand_d,sand_l, texture = False, get_mesh =True,getwire=False, show=False)
        translation_vector = np.array([sand_d.shape[1]+20, 0.0, 0.0]) 
        shell_mesh.translate(translation_vector)

        shell_pointcloud = DepthPipeline.Get3DVis(shell_d,shell_l, texture = True, get_mesh =False,getwire=False, show=False)
        sand_pointcloud = DepthPipeline.Get3DVis(sand_d,sand_l, texture = True, get_mesh =False,getwire=False, show=False)
        translation_vector = np.array([sand_d.shape[1]+20, 0.0, 0.0]) 
        shell_pointcloud.translate(translation_vector)
        translation_vector = np.array([0.0, 0.0, 100]) 
        shell_pointcloud.translate(translation_vector)
        sand_pointcloud.translate(translation_vector)

        
        mesh_heat, plane = DepthPipeline.ShowFittedPlaneToMesh (np.hstack((sand_d,shell_d)),np.hstack((sand_l,shell_l)), heat_map = True, show_plane = True)
        translation_vector = np.array([0.0, 0.0, -120]) 
        mesh_heat.translate(translation_vector)
        plane.translate(translation_vector)
        mesh_heat_sand, mesh_heat_shell = split_mesh_along_x_with_gap(mesh_heat)

        o3d.visualization.draw_geometries([sand_mesh,shell_mesh,sand_pointcloud, shell_pointcloud,mesh_heat_sand, mesh_heat_shell,plane ],width=1920*2, height=1080*2,window_name="3d surface")

    def compute_psd_features(height_map, high_freq_threshold_ratio=0.25):
        # Remove mean
        height_map_centered = height_map - np.mean(height_map)

        # Apply 2D Hann window to reduce edge artifacts
        window = np.outer(windows.hann(height_map.shape[0]), windows.hann(height_map.shape[1]))
        height_map_windowed = height_map_centered * window

        # Compute 2D FFT and PSD
        fft_result = fftshift(fft2(height_map_windowed))
        psd2D = np.abs(fft_result) ** 2

        # Radial averaging to get 1D PSD
        y, x = np.indices(psd2D.shape)
        center = np.array(psd2D.shape) // 2
        r = np.hypot(x - center[1], y - center[0])
        r = r.astype(np.int32)
        r_max = np.min(center)

        radial_psd = np.bincount(r.ravel(), psd2D.ravel()) / np.bincount(r.ravel())
        freqs = np.arange(len(radial_psd))

        # Spectral slope (log-log linear fit)
        valid = (freqs > 1) & (freqs < r_max)
        log_freqs = np.log(freqs[valid])
        log_psd = np.log(radial_psd[valid])

        def linear(x, a, b): return a * x + b
        beta, _ = curve_fit(linear, log_freqs, log_psd)[0]

        # Total Power
        total_power = np.sum(radial_psd)

        # High-Frequency Ratio
        cutoff = int(high_freq_threshold_ratio * r_max)
        high_freq_ratio = np.sum(radial_psd[cutoff:]) / total_power

        # Spectral Centroid
        spectral_centroid = np.sum(freqs * radial_psd) / np.sum(radial_psd)

        # Spectral Entropy
        psd_norm = radial_psd / np.sum(radial_psd)
        spectral_entropy = -np.sum(psd_norm * np.log(psd_norm + 1e-12))  # avoid log(0)

        return {
            'spectral_slope_beta': -beta,  # sign flipped so that higher = smoother
            'total_power': total_power,
            'high_freq_ratio': high_freq_ratio,
            'spectral_centroid': spectral_centroid,
            'spectral_entropy': spectral_entropy
        },psd2D,radial_psd
    
    shell_frequency_features,shell_psd2D, shell_radial_psd = compute_psd_features(shell_detrened)
    sand_frequency_features, sand_psd2D, sand_radial_psd = compute_psd_features(sand_detrened)
    print("Shell-Hash", "Sand")
    for key in shell_frequency_features.keys():
        print(key+":", shell_frequency_features[key], sand_frequency_features[key])
    fig1, axs1 = plt.subplots(1, 2, figsize=(10, 4))

    axs1[0].imshow(np.log1p(sand_psd2D), cmap='inferno')
    axs1[0].set_title("Sand - Log Power Spectrum", fontsize=14)
    axs1[0].axis('off')

    axs1[1].imshow(np.log1p(shell_psd2D), cmap='inferno')
    axs1[1].set_title("Shell Hash - Log Power Spectrum",fontsize=14)
    axs1[1].axis('off')
    axs1[0].tick_params(axis='both', labelsize=12)
    axs1[1].tick_params(axis='both', labelsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join("dataset", f"psd_maps_{experiment_index}.png"))
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    freqs = np.arange(1, len(sand_radial_psd))  # Skip 0
    ax2.loglog(freqs, sand_radial_psd[1:], label="Sand", color='tab:blue')
    ax2.loglog(freqs, shell_radial_psd[1:], label="Shell-Hash", color='tab:orange')
    ax2.set_title("Radial PSD (Log-Log)", fontsize=14)
    ax2.set_xlabel("Spatial Frequency", fontsize=14)
    ax2.set_ylabel("Power", fontsize=14)
    ax2.legend(fontsize=14)
    ax2.grid(True)
    ax2.tick_params(axis='both', labelsize=12)
    

    plt.tight_layout()
    plt.savefig(os.path.join("dataset", f"psd_radial_{experiment_index}.png"))


def DoBaselineExperiment():
    DepthPipeline = DepthMono(RaftModelPath = os.path.join("raft_models", "raftstereo-eth3d.pth"),
                            OutPutSize = (640,480),
                            PreProcessFlag=True,
                            ROI= None,
                            BorderOffset = 50,
                            ScaleDisps =10,
                            SaveFlag=False,
                            SavePath = "dataset",
                            ShowScale=50,
                            Debug = False)
    ROIs = [[(1021, 1226), (2913, 2533)],[(1056, 1193), (2970, 2576)]]
    shell1_left_path = os.path.join("dataset","7b",str(8), "left.png")
    shell1_right_path = os.path.join("dataset","7b" ,str(8), "right.png")
    shell1_left = cv2.imread(shell1_left_path)
    shell1_right = cv2.imread(shell1_right_path)
    shell2_left_path = os.path.join("dataset","7b" ,str(23), "left.png")
    shell2_right_path = os.path.join("dataset","7b" ,str(23), "right.png")
    shell2_left = cv2.imread(shell2_left_path)
    shell2_right = cv2.imread(shell2_right_path)
    DepthPipeline.ROI = ROIs[1]
    
    

    rleft,rright = DepthPipeline.GetRectifiedImages(shell1_left,shell1_right)
    shell1_l,shell1_d = DepthPipeline.GetDispMap(rleft, rright)
    shell1_features, shell1_detrened = DepthPipeline.GetSurfaceStats(shell1_d)
    DepthPipeline.Reset()
    DepthPipeline.ROI = ROIs[1]
    rleft,rright = DepthPipeline.GetRectifiedImages(shell2_left,shell2_right)
    shell2_l,shell2_d = DepthPipeline.GetDispMap(rleft, rright)
    shell2_features,shell2_detrened = DepthPipeline.GetSurfaceStats(shell2_d)
    DepthPipeline.ROI = None
    all_features = ["Sa", "Sq", "Mp", "sigma_p", "Mt", "sigma_t",
                "skew", "kurtosis", "sk", "spk", "svk", 
                "ratio_spk_sk", "ratio_svk_sk"]

    # Indices of stable features
    keep_indices = [0, 1, 2, 3,4,5,6,8]  
    features_names = [all_features[i] for i in keep_indices]

    arr1 = np.asarray(shell1_features)[keep_indices]
    arr2 = np.asarray(shell2_features)[keep_indices]

    plt.figure(figsize=(6, 4))
    plt.plot(arr1, marker='o', label='Shell-Hash-Experiment-1',c="green")
    plt.plot(arr2, marker='s', label='Shell-Hash-Experiment-3',c="brown")
    plt.ylabel("Value", fontsize = 12)
    plt.xlabel("Microtopography Features", fontsize = 12)
    plt.xticks(ticks=np.arange(len(features_names)), labels=features_names, rotation=0)
    plt.legend(fontsize = 12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join("dataset", f"baseline-experiment.png"))
    print("results saved in "+os.path.join("dataset", f"baseline-experiment.png"))







if __name__ == "__main__":

    # save location map ################################
    # Location Sand
    lat_a = -36.16922
    lon_a = +175.103999
    # Location Shell-Hash
    lat_b = -36.2365 
    lon_b = 175.0361667
    center_lat = (lat_a + lat_b) / 2
    center_lon = (lon_a + lon_b) / 2
    m = folium.Map(location=[center_lat, center_lon], zoom_start=14)
    folium.Marker([lat_a, lon_a], popup="Location A (Sand)", icon=folium.Icon(color='blue')).add_to(m)
    folium.Marker([lat_b, lon_b], popup="Location B (Shell-Hash)", icon=folium.Icon(color='red')).add_to(m)
    folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
    attr="© Esri",
    name="ESRI Light Gray Canvas"
    ).add_to(m)
    folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr="© Esri",
    name="ESRI Satellite",
    overlay=False,
    control=True
    ).add_to(m)
    folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}",
    attr="© Esri",
    name="Labels & Boundaries",
    overlay=True,
    control=True
    ).add_to(m)
    folium.LayerControl().add_to(m)
    html_path = "dataset/sediment_sites_map.html"
    m.save(html_path)
    ################################################################

    #sand       7, 23, 19
    #shell-hash 8, 10, 23

    # Choose from these pairs to get the same results as in the paper
    # # Experiment 1
    # experiment = (7,8)
    # experiment_index = 1

    # Experiment 2
    # experiment = (23,10)
    # experiment_index = 2

    # Experiment 3
    experiment = (19,23)
    experiment_index = 3

    # DoExperimentAll(experiment,experiment_index, show3d = False)
    DoBaselineExperiment()








# The following script was used to get pair of frames manually from two sampled videos
# The results are in the dataset folder
# def GetPairs():
#     pari_count = 0
#     left = None
#     right = None
#     left_flag = False
#     image_folder = "dataset/7b"
#     out = "dataset/7b/pairs"
#     if not os.path.exists(out):
#         os.makedirs(out)
#     image_files = sorted([
#         f for f in os.listdir(image_folder)
#         if f.lower().endswith(('.png', '.jpg', '.jpeg'))
#     ])
#     idx = 0
#     while idx < len(image_files):
#         image_path = os.path.join(image_folder, image_files[idx])
#         image_org = cv2.imread(image_path)
#         if image_org is None:
#             print(f"Failed to load {image_path}")
#             continue
#         idx+=1
#         # Resize to half size
#         image = cv2.resize(image_org, (image_org.shape[1] // 4, image_org.shape[0] // 4))

#         # Show image
#         cv2.imshow("Image Viewer", image)
#         key = cv2.waitKey(0) & 0xFF

#         if key== ord('f'):
#             continue

#         if key== ord('a'):
#             left = image_org.copy()
#             left_flag = True
#             continue
#         if key== ord('s'):
#             right = image_org.copy()
#             if left_flag:
#                 outt = os.path.join(out, str(pari_count))
#                 if not os.path.exists(outt):
#                     os.makedirs(outt)
#                 cv2.imwrite(os.path.join(outt,"left.png"), left)
#                 cv2.imwrite(os.path.join(outt,"right.png"), right)
#                 left_flag = False
#                 pari_count+=1
#             continue
#         elif key == ord('q'):
#             break
#     cv2.destroyAllWindows()