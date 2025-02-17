import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import logging 
import os 
import shutil

logger = logging.getLogger(__name__)

class DataPrepration:
    def __init__(self, input_path: str, output_path: str) -> None:
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self._make_directories()

    def _make_directories(self) -> None:
        (self.output_path / "images").mkdir(parents=True, exist_ok=True)
        (self.output_path / "points").mkdir(parents=True, exist_ok=True)
        (self.output_path / "bboxes_3d").mkdir(parents=True, exist_ok=True)
        (self.output_path / "masks").mkdir(parents=True, exist_ok=True)

    def copy_point_cloud(self, points_file: str, file_id: int) -> None:
        outfile = self.output_path / "points" / f"{file_id}.npy"
        shutil.copy(points_file, outfile)

    def copy_bbox3d(self, bbox3d_file: str, file_id: int) -> None:
        outfile = self.output_path / "bboxes_3d" / f"{file_id}.npy"
        shutil.copy(bbox3d_file, outfile)

    def copy_masks(self, mask_file: str, file_id: int) -> None:
        outfile = self.output_path / "masks" / f"{file_id}.npy"
        shutil.copy(mask_file, outfile)

    def copy_images(self, iamge_file: str, file_id: int) -> None:
        outfile = self.output_path / "images" / f"{file_id}.jpg"
        shutil.copy(iamge_file, outfile)

    def run(self):
        data_dirs = [d for d in self.input_path.iterdir() if d.is_dir()]
        for idx, data_dir in enumerate(tqdm(data_dirs, desc="Processing Data"), start=1):
            file_id = f"{idx:05d}"
            points_file = data_dir / "pc.npy"
            self.copy_point_cloud(points_file, file_id)
            bbox3d_file = data_dir / "bbox3d.npy"
            self.copy_bbox3d(bbox3d_file, file_id)
            mask_file = data_dir / "mask.npy"
            self.copy_masks(mask_file, file_id)
            image_file = data_dir / "rgb.jpg"
            self.copy_images(image_file, file_id)


def main():
    parser = argparse.ArgumentParser(description="Data Preparation")
    parser.add_argument("--input-path", type=str, required=True, help="Input raw data directory path")
    parser.add_argument("--output-path", type=str, required=True, help="Output customized data directory path")

    args = parser.parse_args()

    # Run data processing ....
    data_preparation = DataPrepration(args.input_path, args.output_path)
    data_preparation.run()


if __name__ == "__main__":
    main()











