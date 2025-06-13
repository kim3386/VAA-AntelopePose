import json
import mmcv
import numpy as np
import cv2  # Import OpenCV for drawing
from mmpose.visualization import PoseLocalVisualizer
from mmpose.structures import PoseDataSample
from mmengine.structures import InstanceData

annotation_file = 'filtered_ap10k_test_antelopes/annotations/test_annotations.json'
image_path = 'filtered_ap10k_test_antelopes/data/000000000163.jpg'

with open(annotation_file, 'r') as f:
    annotations = json.load(f)

#Find annotation
image_id = next((img['id'] for img in annotations['images'] if img['file_name'] == image_path.split('/')[-1]), None)

if image_id is None:
    raise ValueError("Image not found in annotation file!")

#Get keypoints
keypoints = [
    {'keypoints': np.array(ann['keypoints']).reshape(-1, 3), 'bbox': ann['bbox']}
    for ann in annotations['annotations'] if ann['image_id'] == image_id
]

#Load image
image = mmcv.imread(image_path)

#Convert keypoints to InstanceData
instance_data = InstanceData()
instance_data.keypoints = np.array([kp['keypoints'] for kp in keypoints])

#Prepare pose data for visualization
pose_sample = PoseDataSample()
pose_sample.gt_instances = instance_data

#Initialize Pose Visualizer
visualizer = PoseLocalVisualizer()

#Draw keypoints using OpenCV
def draw_keypoints_custom(image, keypoints, border_color=(0, 0, 0), keypoint_color=(0, 0, 255), border_thickness=2, keypoint_thickness=2):
    for kpt in keypoints:
        for x, y, confidence in kpt:
            if confidence > 0:
                cv2.circle(image, (int(x), int(y)), 3, border_color, border_thickness)
                cv2.circle(image, (int(x), int(y)), 1, keypoint_color, keypoint_thickness)
    return image

visualized_image = image.copy()
for kpt in instance_data.keypoints:
    visualized_image = draw_keypoints_custom(visualized_image, [kpt])

mmcv.imwrite(visualized_image, 'outputs/000000000163_annotate.jpg')
