import argparse
import numpy as np
import matplotlib.pyplot as plt


def visualize_point_cloud(point_cloud):
        
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    channels = ['X', 'Y', 'Z']
    
    for i, (ax, channel) in enumerate(zip(axes, channels)):
        channel_data = point_cloud[:, i]
        channel_data = channel_data.reshape(-1, 1)  
        
        im = ax.imshow(channel_data, cmap="jet", aspect='auto')
        ax.set_title(f'{channel} Channel', fontsize=12)
        fig.colorbar(im, ax=ax, shrink=0.8)
    
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Point Cloud Channels Visualization')
    parser.add_argument('file_path', type=str, help='point cloud file')
    args = parser.parse_args()
    
    point_cloud = np.load(args.file_path)
    visualize_point_cloud(point_cloud)

if __name__ == "__main__":
    main()