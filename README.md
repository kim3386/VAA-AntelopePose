

# Project Overview

From our [Abstract](weekly_meetings/Resources.pdf):

Our research aims to increase the robustness and efficiency of antelope pose estimation models through refining the ground-truth labels of keypoints and compressing the training dataset by filtering for similar species.

We focus on AP-10K, a prominent animal pose dataset, which we observe to have inconsistent keypoint definitions across images, hindering model performance. To address this, we create two distinct keypoint definitions. First, in an effort to have more consistently labeled images, we create a “Visible” keypoint definition, in which we choose the most visibly apparent features, enabling easier and more consistent labeling. Second, we focus on the biologically accurate point for keypoints, which would be harder for labelers, but could allow the model to better generalize on the features.

We utilize three categories of methods to focus an off-theshelf pose estimation model’s training data to only include species morphologically similar to Antelopes: handcrafted measures (the normalized variance of each keypoint’s distance from the animals’ centroid and normalized limb lengths), traditional feature extraction (ORB), and deep learning features (extracted by Meta’s DINOv2 transformer model.

Through our efforts to improve model performance in keypoint estimation for specifically Antelopes, we contribute to improving keypoint estimation for all animals.

[Keypoint Definitions](weekly_meetings/Resources.pdf)


# Weekly Meetings
[Meeting Template](weekly_meetings/VAA-weekly-meeting-template-v240820.pptx)

[Resources and Links](weekly_meetings/Resources.pdf)


# Troubleshooting Tips

For returning members to update the new repo remote:
git remote set-url origin --push --add git@github.com:VIP-VAA-S25/MD_Doc.git

### E.g., mmpose installation 
instead of doing `mim install "mmcv>=2.0.1"`
do `mim install "mmcv==2.1.0"`

Reason: mmdet is not compatible with the latest version of mmcv


# Useful command lines

MMPose:
1. HRNet, trained on AP-10k
`python demo/image_demo.py sample_images/Springboks.webp configs/animal_2d_keypoint/topdown_heatmap/ap10k/td-hm_hrnet-w48_8xb64-210e_ap10k-256x256.py hrnet_w48_ap10k_256x256-d95ab412_20211029.pth --out-file ./sample_images/output3-ap10k.jpg --draw-heatmap`
2. RTMPose, training on AP-10K:

   Activate conda environment: `openmmlab`

   Get into `~/mmpose` directory

   `python /home/vip24_shared/mmpose/tools/train.py /home/vip24_shared/mmpose/configs/animal_2d_keypoint/rtmpose/ap10k/rtmpose-m_8xb64-210e_ap10k-256x256.py`

4. Training on subset:
` python /home/vip24_shared/mmpose/tools/train.py /home/vip24_shared/mmpose/filtered_ap10k_antelopes/rtmpose-m_8xb64-210e_ap10k-256x256-subset.py`
5. Testing example:
`python tools/test.py /home/vip24_shared/mmpose/filtered_ap10k_bovidae+cervidae/rtmpose-m_8xb64-210e_ap10k-256x256_b+c.py /home/vip24_shared/mmpose/training_logs2/bovidae_cervidae_run1/epoch_210.pth`
6. Inferencing example:
   `python /home/vip24_shared/mmpose/demo/inferencer_demo.py \
--pose2d /home/vip24_shared/mmpose/filtered_ap10k_bovidae+cervidae/rtmpose-m_8xb64-210e_ap10k-256x256_b+c.py \
--pose2d-weights /home/vip24_shared/mmpose/training_logs2/bovidae_cervidae_run1/epoch_210.pth \
--vis-out-dir /home/vip24_shared/mmpose/sample_images/output_b+c/ \
--device cpu \
--draw-bbox \
/home/vip24_shared/mmpose/sample_images/000000000001.jpg
`



# Data Format

`import json`

`f = open('/home/vip24_shared/mmpose/data/ap10k/annotations/ap10k-train-split1.json')`

`data = json.load(f)`
 
- AP-10K
	- dict_keys(['info', 'licenses', 'images', 'annotations', 'categories'])
	- data['info']
	  {'description': 'AP-10k', 'url': 'https://github.com/AlexTheBad/AP-10K', 'version': '1.0', 'year': 2021, 'contributor': 'AP-10k Team', 'date_created': '2021/07/01'}
	- data['images'][0]
	  {'license': 1, 'id': 102, 'file_name': '000000000102.jpg', 'width': 1024, 'height': 681, 'background': 5}
	- data['annotations'][0]
	  {'id': 114, 'image_id': 102, 'category_id': 1, 'bbox': [424, 205, 552, 456], 'area': 251712, 'iscrowd': 0, 'num_keypoints': 10, 'keypoints': [885, 355, 2, 810, 361, 2, 843, 412, 2, 864, 529, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 729, 571, 2, 822, 604, 2, 678, 619, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 590, 473, 2, 456, 622, 2, 549, 631, 2]}
	- data['categories'][0]
	  {'id': 1, 'name': 'antelope', 'supercategory': 'Bovidae', 'keypoints': ['left_eye', 'right_eye', 'nose', 'neck', 'root_of_tail', 'left_shoulder', 'left_elbow', 'left_front_paw', 'right_shoulder', 'right_elbow', 'right_front_paw', 'left_hip', 'left_knee', 'left_back_paw', 'right_hip', 'right_knee', 'right_back_paw'], 'skeleton': [[1, 2], [1, 3], [2, 3], [3, 4], [4, 5], [4, 6], [6, 7], [7, 8], [4, 9], [9, 10], [10, 11], [5, 12], [12, 13], [13, 14], [5, 15], [15, 16], [16, 17]]}
- keypoints: 51-dim, each 3 digits represent a keypoint

