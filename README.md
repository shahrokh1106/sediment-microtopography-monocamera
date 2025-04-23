# Zero-Shot Seafloor Sediment Microtopography Characterization Using Stereo from a Drifting Monocular Camera
High-resolution characterization of seafloor sediment microtopography is essential for understanding benthic habitat structure, sedimentary processes, and ecological function. However, existing methods typically rely on core-based sampling or specialized 3D imaging systems, both of which are limited by cost, complexity, and scalability. In this study, we present a cost-effective, camera-based framework for quantitative sediment surface analysis using video footage from a drifting monocular underwater camera. The method leverages a zero-shot application of RAFT-Stereo to estimate dense disparity maps from sequential frames without requiring prior training on sediment data. After normalizing disparities, we apply surface detrending and extract statistical and morphological roughness features. Through a small-scale case study, we evaluate the method on two distinct sediment types, Sand and Shell-Hash, and demonstrate that the extracted features effectively capture surface complexity. Additionally, we assess the pipeline’s consistency using overlapping samples acquired with different virtual stereo baselines, showing that key global features remain robust despite variations in camera motion. This framework offers a scalable, non-invasive solution for retrospective and in-situ sediment analysis in marine monitoring.

The outline comprises six main components: camera calibration, data acquisition, distortion removal, preprocessing, depth estimation, and microtopography feature extraction. 
![outline](https://github.com/shahrokh1106/sediment-microtopography-monocamera/figs/outline.png)

Stereo vision typically involves two cameras positioned at slightly different angles, mimicking human vision, to perceive depth. However, in our scenario, a single moving camera is used. The approach involves capturing a sequence of two frames, treating the first frame as the "left stereo image” and the subsequent frame, obtained after a short interval depending on camera speed, as the "right stereo image” in a simulated stereo system. This simulates the displacement of viewpoints like a stereo camera setup. In this approach, the displacement between the two sequential video frames is a basis for estimating depth. However, it's important to note that inconsistencies in camera speed across the video sequence can lead to variations in the perceived baseline (the distance between virtual camera poses) for each pair of stereo images. These inconsistencies can introduce errors in depth estimation. Nevertheless, the primary objective of this method is not to achieve precise depth measurements but to assess roughness by identifying surface irregularities. The focus lies on characterizing microtopography variations for comparative analysis between different sites. 

![example](https://github.com/shahrokh1106/sediment-microtopography-monocamera/assets/44213732/710d23fa-c522-4bc1-9385-cae019b36b09)

The Python files *DepthMono.py* and *utilities.py* implement the necessary steps to get depths from sequential frames. In *DepthMono.py*

* *OutPutSize* is the size of input images after distortion removal and preprocessing steps. If SuperGlue model is used for the feature matching,  *OutPutSize* must be (640,480)
* *PreProcessFlag* is a flag to set whether preprocessing is needed. If True, Zero-cross normalization followed by Adaptive Histogram Equalization will be applied to the input images.
* *DistCoef* is the distortion parameters estimated from the calibration part. The initial values in the code are related to the drop camera we used.
* *CameraMatrix* is the camera matrix estimated from the calibration step
* *MatchingModel* loads the feature matching model (SuperGlue)
* *DispModelPath* is the disparity estimation model path; we use CREStere model

After getting disparity maps from the input video, the conversion to the depth map is trivial, using the calibration parameters. Next, microtopography surface analysis is applied to the depth maps using detrending approach where a plane is fitted to through the depth surfaces. The below shows an example. 

![detrending](https://github.com/shahrokh1106/sediment-microtopography-monocamera/assets/44213732/6c06109c-d1ee-43bb-8b74-91b44dfd5339)



