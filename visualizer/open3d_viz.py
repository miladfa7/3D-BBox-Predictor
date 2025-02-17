import numpy as np
import os 
from pathlib import Path
import argparse
from typing import List

import open3d as o3d
from open3d.visualization import Visualizer
from open3d import geometry


class Open3DViz:
    def __init__(self, name: str = "Open3D Visualizer", save_dir: Path = "") -> None:
        self.o3d_vis = self._initialize_o3d_viz()

    def _initialize_o3d_viz(self) -> Visualizer:
        """
        Initialize open3d visualizer
        """
        o3d_vis = o3d.visualization.VisualizerWithKeyCallback()
        o3d_vis.create_window()
        o3d_vis.get_view_control()
        return o3d_vis

    def draw_point_cloud_3d(self, 
                            points: np.ndarray,
                            points_size: int =2) -> None:
        if points.size == 0:
            raise ValueError("Point cloud data is empty.")
        
        points = points.transpose(1, 2, 0).reshape(-1, 3)

        o3d_render = self.o3d_vis.get_render_option()
        o3d_render.point_size = points_size
        point_cloud_geo = geometry.PointCloud()
        point_cloud_geo.points = o3d.utility.Vector3dVector(points)
        self.o3d_vis.add_geometry(point_cloud_geo)

    def draw_bboxes_3d(self, 
                       bboxes_3d: np.ndarray, 
                       points_color: List = [1, 0, 0],
                       lines_color: List = [0, 0, 1]) -> None:
        for bbox_3d in bboxes_3d:
            bbox_cloud = geometry.PointCloud() # TODO: It can be optimized
            bbox_cloud.points = o3d.utility.Vector3dVector(bbox_3d)
            bbox_cloud.paint_uniform_color(points_color)

            lines_connection = [
                [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom
                [4, 5], [5, 6], [6, 7], [7, 4],  # Top
                [0, 4], [1, 5], [2, 6], [3, 7]   # Vertical
            ]

            line_set = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(bbox_3d),
                lines=o3d.utility.Vector2iVector(lines_connection)
            )
            line_set.paint_uniform_color(lines_color)

            # self.o3d_vis.add_geometry(bbox_cloud)
            self.o3d_vis.add_geometry(line_set)


    def run(self, show: bool = True, save_image: bool = False) -> None:
        if show:
            self.o3d_vis.run()
        if save_image:
            self.o3d_vis.capture_screen_image("output_image.png")

def parse_args():
    parser = argparse.ArgumentParser(description="Open3D arg parser")
    parser.add_argument("--sample", type=str, default=None, required=True, help="Sample data directory containing 'pc.npy'")
    parser.add_argument("--draw_3d_box", action="store_true", default=False, help="Draw 3D bounding boxes on the point cloud visualization")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    sample_dir = Path(args.sample)
    
    if not os.path.exists(sample_dir):
        raise FileNotFoundError(f"Sample directory '{sample_dir}' does not exist.")
    
    # Load point cloud
    points_file = sample_dir / "pc.npy"
    points = np.load(points_file)

    open_3d_viz = Open3DViz()
    open_3d_viz.draw_point_cloud_3d(points)

    if args.draw_3d_box:
        bboxes_file = sample_dir / "bbox3d.npy"
        bboxes = np.load(bboxes_file)
        open_3d_viz.draw_bboxes_3d(bboxes)
    open_3d_viz.run()


if __name__=="__main__":
    main()
