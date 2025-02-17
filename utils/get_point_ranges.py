import os
import numpy as np
from tqdm import tqdm  

def compute_point_cloud_range(data_dir):
  
    global_min = np.array([np.inf, np.inf, np.inf])
    global_max = np.array([-np.inf, -np.inf, -np.inf])
    
    pc_files = [file for file in os.listdir(data_dir)]
    
    for filename in tqdm(pc_files):
        filepath = os.path.join(data_dir, filename)       
        points = np.load(filepath)
        points = points.reshape(3, -1)
        xyz = points[:, :3]
        
        current_min = np.min(xyz, axis=0)
        current_max = np.max(xyz, axis=0)
        
        global_min = np.minimum(global_min, current_min)
        global_max = np.maximum(global_max, current_max)
    
    point_cloud_range = np.concatenate([global_min, global_max]).tolist()
    
    return point_cloud_range

data_directory = "../dataset/points"
pc_range = compute_point_cloud_range(data_directory)
print("Computed Point Cloud Range:", pc_range)

