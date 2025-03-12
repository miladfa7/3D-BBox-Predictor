import argparse
import torch
from models.PointPillars.pointpillars.model import PointPillars
import numpy as np
from utils import center_to_corner_box3d




class Sereact3DBoxTest:
    def __init__(self, model_path: str) -> None:
        self.points_range = [-0.94, -0.93, -0.93, 2.80, 2.8, 2.8]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pointpillars = PointPillars(nclasses=1, voxel_size=[0.01, 0.01, 0.01],
                                         point_cloud_range=[-0.94, -0.93, -0.93, 2.80, 2.8, 2.8],
                                         max_num_points=32,
                                         max_voxels=(30000, 50000)).to(self.device)

        self.pointpillars.load_state_dict(torch.load(model_path))

    def params_to_corners(self, params: np.ndarray) -> np.ndarray:
        # Extract parameters
        center = params[:, :3]
        size = params[:, 3:6]
        angle = None # No information about rotation angel
        corners = center_to_corner_box3d(center, size, angle)
        return corners

    def convert_boxes_to_corners(self, boxes: np.ndarray) -> np.ndarray:
        pass

    def run(self, pts_path: str) -> None:
        pts = np.load(pts_path)
        pts = pts.astype(np.float32).reshape(3, -1).T
        pts_torch = torch.from_numpy(pts)
        calib_info = None

        self.pointpillars.eval()
        with torch.no_grad():
            pts_torch = pts_torch.float().contiguous().to(self.device)
            batched_pts = [pts_torch]
            predicted_3d_bboxes = self.pointpillars(batched_pts=batched_pts, mode='test')

            bboxes = predicted_3d_bboxes[0]["lidar_bboxes"]
            bboxes = np.expand_dims(bboxes, axis=0)
            bboxes_3d = self.params_to_corners(bboxes)
            print(bboxes_3d)
            # bboxes_3d = np.array(bboxes_3d)
            np.save("test.npy", bboxes_3d)
               




if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Test Sereact3DBox')
    parser.add_argument('--model_path', type=str, default='checkpoints/pointspillars_16.pth',
                        help='Path to the trained model')
    parser.add_argument('--pts_path', type=str, default='/mnt/disk2/users/milad/Research/3D-Bboxes/Pipeline_2/dataset/points/00001.npy',
                        help='Path to the point cloud file')
    args = parser.parse_args()

    test_runner = Sereact3DBoxTest(args.model_path)
    test_runner.run(args.pts_path)