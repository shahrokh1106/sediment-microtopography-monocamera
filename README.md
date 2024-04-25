# Sediment microtopography analysis based on depth estimation from a single moving drop camera
A low-cost solution for deep-sea sediment characteristic analysis by incorporating photogrammetry and AI approaches based on optical imaging and a single-view camera. In contrast to conventional methods reliant on specialized equipment such as laser or stereo setups, our innovative approach offers a distinct advantage in its ability to analyse already recorded single-view videos. Our methodology opens avenues for cost-effective deep-sea sediment characteristic analysis without needing elaborate or expensive gear.


* Dataset: Seabed videos from the subtidal environment around Little Barrier Island
* Calibrating their drop camera
* Feature-based rectification and disparity map computation
* Depth map estimation and 3D-reconstruction from rectified video frames
* Micro-topography characteristic analysis based on depth maps and surface reconstruction

The outline comprises six main components: camera calibration, data acquisition, distortion removal, preprocessing, depth estimation, and microtopography feature extraction. 
![outline](https://github.com/shahrokh1106/sediment-microtopography-monocamera/assets/44213732/06863f29-9cd6-458b-9a2a-da5a4a15fb56)

Stereo vision typically involves two cameras positioned at slightly different angles, mimicking human vision, to perceive depth. However, in our scenario, a single moving camera is used. The approach involves capturing a sequence of two frames, treating the first frame as the "left stereo image” and the subsequent frame, obtained after a short interval depending on camera speed, as the "right stereo image” in a simulated stereo system. This simulates the displacement of viewpoints akin to a stereo camera setup. In this approach, the displacement between the two sequential video frames is a basis for estimating depth. However, it's important to note that inconsistencies in camera speed across the video sequence can lead to variations in the perceived baseline (the distance between virtual camera poses) for each pair of stereo images. These inconsistencies can introduce errors in depth estimation. Nevertheless, the primary objective of this method is not to achieve precise depth measurements but to assess roughness by identifying surface irregularities. The focus lies on characterizing microtopography variations for comparative analysis between different sites. 

![example](https://github.com/shahrokh1106/sediment-microtopography-monocamera/assets/44213732/710d23fa-c522-4bc1-9385-cae019b36b09)

