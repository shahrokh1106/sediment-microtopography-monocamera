from DepthMonoCore import *
def main():
    DispPipeline = DepthMono(RaftModelPath = os.path.join("raft_models", "raftstereo-eth3d.pth"),
                             OutPutSize = (640,480),
                             PreProcessFlag=True,
                             ROI= [(1036, 1280), (3100, 2416)],
                             BorderOffset = 20,
                             ScaleDisps =20,
                             SaveFlag=True,
                             SavePath = "example/sediment_results",
                             ShowScale=50,
                             Debug = True)
    
    left = cv2.imread("example/left.png")
    right = cv2.imread("example/right.png")
    rleft,rright = DispPipeline.GetRectifiedImages(left,right)
    l,d = DispPipeline.GetDispMap(rleft, rright)
    DispPipeline.ShowFittedPlaneToMesh (d,l, heat_map = True, show_plane = True)
    mesh = DispPipeline.Get3DVis(d,l, texture = True, get_mesh =False,getwire=False, show=True)
    DispPipeline.GetSurfaceStats(d)
    peaks, pits = DispPipeline.GetPeaksPits(d,show3d=True)
    print("Done")
if __name__ == "__main__":
    main()
