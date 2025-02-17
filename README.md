  

# Sereact 3D bounding Box Predictor

  

## Code Challenge Implementations

- Data Analysis
- Data Preprocessing
- Models
- Performance Log
- Suggestion
  
### Required Packages: 
```
torch
timm
opencv
open3d
nvidia-cu12.8
matplotlib
wandb
numpy
```

## 1. Data Analysis

### Point Cloud: <br>
Based on my understanding, the point cloud data is organized in a structure similar to an image. Its shape is (3, height, width), where each pixel corresponds to a 3D point with X, Y, and Z coordinates.


**Point Cloud Channels Visualization**.

- X-axis: Horizontal displacement from the camera center (left: negative, right: positive, center: zero)

- Y-axis: Vertical displacement from the camera center (above: negative, below: positive, center: zero)

- Z-axis: Depth distance from the camera, changing smoothly from top to bottom.<br>

```python
python visualizer/point_viz.py --file_path dataset/points/00001.npy
```

<br>
<img src="./images/point_cloud_viz.png" alt="point cloud viz" width="700"> <br>

<br>

**3D boudning Boxes(Ground Truth) Visualization**<br>
3D bounding box represnts 8 corners, each having three values x, y, z. Its shape is (num_objects, 8, 3).

```python
python visualizer/open3d_viz.py --sample 859074c5-9915-11ee-9103-bbb8eae05561 --draw_3d_box
```

<img src="./images/3dviz.png" alt="3D bbox Ground Truth" width="800"> <br>

**Mask and generate 2D Bounding Boxes**.
```python
python visualizer/mask_2dbox_viz.py --image_path ./dataset/images/00026.jpg  --mask_path ./dataset/masks/00026.npy
```
<img src="./images/mask_2dbox.png" alt="mask and 2d bounding box" width="800"> <br>


### 2. Data Preparation and Preprocessing

#### Re-structured raw dataset

Restructuring raw data organizes it into a standardized format, making it easier to process and access.

**Command:**
```python
python prepare_dataset.py --input-path ./raw_data --output-path dataset 
```
**DATASET Structure**


    ```
    dataset/
    │── points/  
    │   ├── 00001.npy  
    │   ├── 00002.npy  
    │   ├── ...  
    │── masks/  
    │   ├── 00001.npy  
    │   ├── 00002.npy  
    │   ├── ...  
    │── bboxes_3d/  
    │   ├── 00001.npy  
    │   ├── 00002.npy  
    │   ├── ...  
    │── images/  
    │   ├── 00001.jpg  
    │   ├── 00002.jpg  
    │   ├── ...  

#### Preprocessing
* **Image (<em>ImageMaskTransforms</em>)**
    - Resizing
    - Normalization
    - To Tensor
    - Augmentations ->  Brightness
* **Point Cloud (<em>PointCloudTransforms</em>)**
    -  Reshape: Converts the points cloud from shape (3, H, W) to (N, 3)for compatibility with the PointNet model or LiDAR-based model
     - Normalization
     - Voxelization
* **3D Boudning Box (<em>BBox3DTransforms</em>)**
    - Reshape: Convert 3D corners representation to centroid, size, orientation(reasoin for model part)
    - Boudning box parameterization: center_x, center_y, center_z, width, height, depth, yaw. its shape is (num_objects, 7)
    - To Tensor
    - Normalization

#### Data Loader
 - **Data Loading:** There are two dataloader such as the <em>**SereactDataLoader**</em>, <em>**PillarsDataLoader**</em> that loads images, point clouds, 3D bounding boxes with support for transformations on each data type.
  - **Data Splitting**: The <em>**DataSpliter**</em> class handles file loading, shuffling, and splitting the all data into training and testing sets. 



## Deep Learning Models
### 1. Point Cloud Based Model (PointPillars)

This model is implemented with the following features:

#### Model Pipeline:
 - Total Parameters: **4,830,140** 
 - Input: **Point Cloud** <br> 
    - Transform the organized point cloud into an unorganized format with shape (N, 3) to feed into the PointNet-based model
 - Voxelization<br>
  Converting 3D point cloud data into a grid of voxels to represent spatial information. The voxel values were specifically tuned for the Sereact dataset point ranges, which were achieved using this script.
    ```
    python utils/get_point_ranges.py
    ```
   - Voxel Size: **[0.01, 0.01, 0.01]**
   - Point Cloud Range: **[-0.94, -0.93, -0.93, 2.80, 2.8, 2.7]**
   - Max Number Points: **16**
   - Max Voxels: **(20000, 35000)**

 - Pillar Feature Encoding: <br>
    - Extract high-level features from each voxel/pillar using a PointNet based model<br>
     - Convert pillars features into densce pseude-images
  - 2D CNN Backbone
     - Process the pseudo-image using 2D CNN layers
  - SSD Detection Head
    - Predict oriented 3D bounding boxes with the output structure  (x,y,z,w,l,h,θ)
    - Utilizes anchor-based regression
 - Output: **3D Bounding Box**
    - Multiple 3D bounding boxes for the point cloud

#### Training Configurations:
```
    Path: configs/pillar_config.yaml
```
 - Loss Function: **Smooth L1**
 - Optimizer: **AdamW**
 - Learning Rate: **0.001**
 - Batch Size: **8**
 - Train and Test Sets: **80%, 20% respectively**
 - Epochs: 

<img src="./images/pointnet.jpg" alt="PointPillars" width="800" height="300"> <br>

#### Run the Model Training 


```
    python train_pillars.py --config configs/pillars_config.yaml
```


### 2. Multi-Modal Model (CNN/ViT & PointNet++) 

This model is implemented with the following features:

#### Model Pipeline:
 - Total Parameters: 
    - ResNet50 + PointNet: **28,476,833**
    - ViT + PointNet: **103,352,417** 
 - Input: **Point Cloud and RGB Image** <br> 
    - Points are retained the raw points without voxelization.normalized to the range [-1, +1], and 100,000 points are sampled from each point cloud.
     - RGB images are normalized to the range [-1, +1], resized to [224, 224], and undergo brightness augmentation.
 - Image Feature Extraction
    - CNN-based -> ResNet50
    - Transformer-Based -> Visual Transformer

 - Point Cloud Features Extraction (PointNet++)
     - Processes raw point clouds to extract spatial features.
 - Fusion Network
    - Combining the extracted features from both the RGB image (CNN output) and the point cloud (PointNet++ output)
 - MLP Layer and Regression Head
    - Processes the fused feature vector to predict 3D bounding boxes.
 - Output: **3D Bounding Box**
    - Multiple 3D bounding boxes using the point cloud and rgb image 

<img src="./images/Multimodal.jpg" alt="Multimodal" width="800" height="370"> <br>


#### Training Configurations:
```
    Path: configs/multimodal_config.yaml
```
 - Loss Function: **Smooth L1**
 - Optimizer: **AdamW**
 - Learning Rate: **0.0001**
 - Batch Size: **8**
 - Train and Test Sets: **80%, 20% respectively**
 - Image Backbone: **Resnet50 or ViT**
 - Point Backbone: **PointNet++**
 - Epochs: **80**


#### Run the Model Training

```python
python3 train_multimodal.py --config configs/multimodal_config.yaml
```

## Model Performance Evaluation with Smooth L1 Loss:

### 1. PointNet-based model performances

#### Logged by WandB


#### Logged by terminal


### 2. MultiModal model performances
#### Logged by WandB

<img src="./images/loss_multimodal.png" alt="Multimodal" width="800" height="370"> <br>


#### Logged by terminal
```
Epoch 1 
========== Train Loss: 1.7952 
========== Eval Loss: 0.3506
Evaluating: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:00<00:00,  6.66it/s]
Epoch 2 
========== Train Loss: 1.0478 
========== Eval Loss: 0.1832
Evaluating: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:00<00:00,  6.70it/s]
Epoch 3 
..
..
..
Evaluating: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:00<00:00,  6.44it/s]
Epoch 75 

Epoch 78 
========== Train Loss: 0.3594 
========== Eval Loss: 0.1036
Evaluating: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:00<00:00,  6.49it/s]
Epoch 79 
========== Train Loss: 0.3628 
========== Eval Loss: 0.1040
Evaluating: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:00<00:00,  6.50it/s]
Epoch 80 
========== Train Loss: 0.3761 
========== Eval Loss: 0.1031
```