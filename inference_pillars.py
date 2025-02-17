import argparse
import torch
from models.PointPillars.pointpillars.model import PointPillars
import numpy as np


def params_to_corners(params: np.ndarray) -> np.ndarray:
    # Extract parameters
    center = params[:3]
    size = params[3:6]
    yaw = params[6]
    l, w, h = size  
    corners_local = np.array([
        [-l/2, -w/2, -h/2],  
        [ l/2, -w/2, -h/2],  
        [ l/2,  w/2, -h/2],  
        [-l/2,  w/2, -h/2],  
        [-l/2, -w/2,  h/2],  
        [ l/2, -w/2,  h/2],  
        [ l/2,  w/2,  h/2],  
        [-l/2,  w/2,  h/2],
    ])
    corners = corners_local + center
    return corners


class Sereact3DBoxTest:
    def __init__(self, model_path: str) -> None:
        self.points_range = [-0.94, -0.93, -0.93, 2.80, 2.8, 2.8]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pointpillars = PointPillars(nclasses=1, voxel_size=[0.01, 0.01, 0.01],
                                         point_cloud_range=[-0.94, -0.93, -0.93, 2.80, 2.8, 2.8],
                                         max_num_points=32,
                                         max_voxels=(20000, 40000)).to(self.device)
        self.pointpillars.load_state_dict(torch.load(model_path))

    def points_range_filter(self, pts: np.ndarray) -> np.ndarray:
        points_range = self.points_range
        flag_x_low = pts[:, 0] > points_range[0]
        flag_y_low = pts[:, 1] > points_range[1]
        flag_z_low = pts[:, 2] > points_range[2]
        flag_x_high = pts[:, 0] < points_range[3]
        flag_y_high = pts[:, 1] < points_range[4]
        flag_z_high = pts[:, 2] < points_range[5]
        keep_mask = flag_x_low & flag_y_low & flag_z_low & flag_x_high & flag_y_high & flag_z_high
        pts = pts[keep_mask]
        return pts 

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
            print(predicted_3d_bboxes)

            bboxes = predicted_3d_bboxes[0]["lidar_bboxes"]
            bboxes_3d = []
            for box in bboxes:
                corners = params_to_corners(box)
                print(corners)
                bboxes_3d.append(corners)
            bboxes_3d = np.array(bboxes_3d)
            np.save("test.npy", bboxes_3d)
               




if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Test Sereact3DBox')
    parser.add_argument('--model_path', type=str, default='checkpoints/pointspillars_15.pth',
                        help='Path to the trained model')
    parser.add_argument('--pts_path', type=str, default='./dataset/points/00001.npy',
                        help='Path to the point cloud file')
    args = parser.parse_args()

    test_runner = Sereact3DBoxTest(args.model_path)
    test_runner.run(args.pts_path)