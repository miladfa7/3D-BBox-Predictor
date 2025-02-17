import numpy as np
from glob import glob

def corners_to_params(corners: np.ndarray):
    min_xyz = np.min(corners, axis=1)
    max_xyz = np.max(corners, axis=1)
    sizes = max_xyz - min_xyz 
    
    return sizes[:, 0], sizes[:, 1], sizes[:, 2]  

if __name__ == "__main__":
    data_dir = "dataset/bboxes_3d"
    
    X, Y, Z = [], [], []
    
    for file in glob(f"{data_dir}/*.npy"):
        bboxes = np.load(file) 
        x_sizes, y_sizes, z_sizes = corners_to_params(bboxes)
        X.extend(x_sizes)
        Y.extend(y_sizes)
        Z.extend(z_sizes)
       

    X, Y, Z = np.array(X), np.array(Y), np.array(Z)

    print("Min X: ", np.min(X), "Max X: ", np.max(X), "Mean X: ", np.mean(X))
    print("Min Y: ", np.min(Y), "Max Y: ", np.max(Y), "Mean Y: ", np.mean(Y))
    print("Min Z: ", np.min(Z), "Max Z: ", np.max(Z), "Mean Z: ", np.mean(Z))
