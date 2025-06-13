import json
import os

ap10k_annotation_files = [ "/home/vip24_shared/mmpose/visible_keypoints/ap10k_matching_annotations/train_ap10k_annotations.json",
             "/home/vip24_shared/mmpose/visible_keypoints/ap10k_matching_annotations/val_ap10k_annotations.json",
             "/home/vip24_shared/mmpose/visible_keypoints/ap10k_matching_annotations/test_ap10k_annotations.json"]
visible_annotation_files = [
            "/home/vip24_shared/mmpose/visible_keypoints/visible_annotations/train_visible_annotations.json",
             "/home/vip24_shared/mmpose/visible_keypoints/visible_annotations/val_visible_annotations.json",
            "/home/vip24_shared/mmpose/visible_keypoints/visible_annotations/test_visible_annotations.json"
]
"""output_files = [
    "/home/vip24_shared/mmpose/visible_keypoints/excluded_annotations/train_eyes_nose.json",
    "/home/vip24_shared/mmpose/visible_keypoints/excluded_annotations/val_eyes_nose.json",
    "/home/vip24_shared/mmpose/visible_keypoints/excluded_annotations/test_eyes_nose.json"
]"""
"""output_files = [
    "/home/vip24_shared/mmpose/visible_keypoints/excluded_annotations/train_neck.json",
    "/home/vip24_shared/mmpose/visible_keypoints/excluded_annotations/val_neck.json",
    "/home/vip24_shared/mmpose/visible_keypoints/excluded_annotations/test_neck.json"
]"""
"""output_files = [
    "/home/vip24_shared/mmpose/visible_keypoints/excluded_annotations/train_tail.json",
    "/home/vip24_shared/mmpose/visible_keypoints/excluded_annotations/val_tail.json",
    "/home/vip24_shared/mmpose/visible_keypoints/excluded_annotations/test_tail.json"
]"""
"""output_files = [
    "/home/vip24_shared/mmpose/visible_keypoints/excluded_annotations/train_hips.json",
    "/home/vip24_shared/mmpose/visible_keypoints/excluded_annotations/val_hips.json",
    "/home/vip24_shared/mmpose/visible_keypoints/excluded_annotations/test_hips.json"
]"""
"""output_files = [
    "/home/vip24_shared/mmpose/visible_keypoints/excluded_annotations/train_legs.json",
    "/home/vip24_shared/mmpose/visible_keypoints/excluded_annotations/val_legs.json",
    "/home/vip24_shared/mmpose/visible_keypoints/excluded_annotations/test_legs.json"
]"""
output_files = [
    "/home/vip24_shared/mmpose/visible_keypoints/excluded_annotations/train_paws.json",
    "/home/vip24_shared/mmpose/visible_keypoints/excluded_annotations/val_paws.json",
    "/home/vip24_shared/mmpose/visible_keypoints/excluded_annotations/test_paws.json"
]
excluded_dir = "/home/vip24_shared/mmpose/visible_keypoints/excluded_annotations"
os.makedirs(excluded_dir, exist_ok=True)

#left eye 0, right eye 1, nose 2
#keypoints_to_replace = [0, 1, 2]

#neck 3
#keypoints_to_replace = [3]

#tail 4
#keypoints_to_replace = [4]

#left hip 11, right hip 14
#keypoints_to_replace = [11, 14]

# left shoulder 5, left elbow 6, right shoulder 8, right elbow 9, left knee 12, right knee 15
#keypoints_to_replace = [5, 6, 8, 9, 12, 15]

#left front paw 7, right front paw 10, left back paw 13, right back paw 16
keypoints_to_replace = [7, 10, 13, 16]

for i in range(len(visible_annotation_files)):
    with open(visible_annotation_files[i], "r") as f:
        visible_data = json.load(f)

    with open(ap10k_annotation_files[i], "r") as f:
        ap10k_data = json.load(f)

    #dictionary by image id
    ap10k_map = {ann["image_id"]: ann for ann in ap10k_data["annotations"]}

    updated = 0
    for ann in visible_data["annotations"]:
        image_id = ann["image_id"]
        if image_id in ap10k_map:
            ap_ann = ap10k_map[image_id]
            for kp_idx in keypoints_to_replace:
                for offset in range(3):  # x, y, visibility
                    ann["keypoints"][kp_idx * 3 + offset] = ap_ann["keypoints"][kp_idx * 3 + offset]
            updated += 1

    with open(output_files[i], "w") as f:
        json.dump(visible_data, f, indent=2)

    print(f"{updated} annotations updated in: {output_files[i]}")
