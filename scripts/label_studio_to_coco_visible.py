import json
import random
#NOT DONE YET - WILL FINISH UP ON SUNDAY
def keypoint_midpoint(keypoint_1, keypoint_2, label):
    """Calculate the midpoint of two keypoints or return a single keypoint if one is missing."""
    if keypoint_1 and keypoint_2:
        return {
            "x": (keypoint_1["x"] + keypoint_2["x"]) / 2,
            "y": (keypoint_1["y"] + keypoint_2["y"]) / 2,
            "keypointlabels": [label],
            "original_width": keypoint_1["original_width"],
            "original_height": keypoint_1["original_height"]
        }
    elif keypoint_1:
        return {
            "x": keypoint_1["x"],
            "y": keypoint_1["y"],
            "keypointlabels": [label],
            "original_width": keypoint_1["original_width"],
            "original_height": keypoint_1["original_height"]
        }
    elif keypoint_2:
        return {
            "x": keypoint_2["x"],
            "y": keypoint_2["y"],
            "keypointlabels": [label],
            "original_width": keypoint_2["original_width"],
            "original_height": keypoint_2["original_height"]
        }
    else:
        return None

def midpoint_consolidation():
    # Load JSON file
    with open('/home/vip24_shared/mmpose/visible_keypoints/project-12-at-2025-03-29-19-26-8c122396.json', 'r') as file:
        data = json.load(file)

    # Process data and calculate midpoints
    for annotation in data:
        keypoints = annotation["kp"]
        
        # Initialize variables
        hip_top, left_hip_bottom, right_hip_bottom = None, None, None
        right_shoulder_outer, right_shoulder_inner = None, None
        left_shoulder_outer, left_shoulder_inner = None, None
        right_back_knee_inner, right_back_knee_outer = None, None
        left_back_knee_inner, left_back_knee_outer = None, None
        neck_top, neck_bottom = None, None

        # Find keypoints
        for kp in keypoints:
            labels = kp.get("keypointlabels", [])
            
            # Hip-related keypoints
            if "hip_top" in labels:
                hip_top = kp
            elif "left_hip_bottom" in labels:
                left_hip_bottom = kp
            elif "right_hip_bottom" in labels:
                right_hip_bottom = kp
            
            # Right shoulder keypoints
            elif "right_shoulder_outer" in labels:
                right_shoulder_outer = kp
            elif "right_shoulder_inner" in labels:
                right_shoulder_inner = kp
            
            # Left shoulder keypoints
            elif "left_shoulder_outer" in labels:
                left_shoulder_outer = kp
            elif "left_shoulder_inner" in labels:
                left_shoulder_inner = kp
            
            # Right back knee keypoints
            elif "right_back_knee_inner" in labels:
                right_back_knee_inner = kp
            elif "right_back_knee_outer" in labels:
                right_back_knee_outer = kp
            
            # Left back knee keypoints
            elif "left_back_knee_inner" in labels:
                left_back_knee_inner = kp
            elif "left_back_knee_outer" in labels:
                left_back_knee_outer = kp
            
            # Neck keypoints
            elif "neck_top" in labels:
                neck_top = kp
            elif "neck_bottom" in labels:
                neck_bottom = kp
        
        # Calculate and append midpoints for various body parts
        #_______________HIP_______________________________
        right_hip, left_hip = None, None
        if right_hip_bottom:
            right_hip = keypoint_midpoint(right_hip_bottom, hip_top, "right_hip")
        if left_hip_bottom:
            left_hip = keypoint_midpoint(left_hip_bottom, hip_top, "left_hip")
        if right_hip:
            keypoints.append(right_hip)
        if left_hip:
            keypoints.append(left_hip)
        
        #___________________Right Shoulder_________________
        right_shoulder = keypoint_midpoint(right_shoulder_inner, right_shoulder_outer, "right_shoulder")
        if right_shoulder:
            keypoints.append(right_shoulder)
        
        #___________________Left Shoulder__________________
        left_shoulder = keypoint_midpoint(left_shoulder_inner, left_shoulder_outer, "left_shoulder")
        if left_shoulder:
            keypoints.append(left_shoulder)
        
        #__________________Right Back Knee_________________
        right_back_knee = keypoint_midpoint(right_back_knee_inner, right_back_knee_outer, "right_knee")
        if right_back_knee:
            keypoints.append(right_back_knee)
        
        #__________________Left Back Knee__________________
        left_back_knee = keypoint_midpoint(left_back_knee_inner, left_back_knee_outer, "left_knee")
        if left_back_knee:
            keypoints.append(left_back_knee)
        
        #_______________________Neck_______________________
        neck = keypoint_midpoint(neck_top, neck_bottom, "neck")
        if neck:
            keypoints.append(neck)

        # Remove old points for cleanliness
        keypoints = [
            kp for kp in keypoints 
            if "hip_top" not in kp.get("keypointlabels", []) 
            and "left_hip_bottom" not in kp.get("keypointlabels", []) 
            and "right_hip_bottom" not in kp.get("keypointlabels", [])
            and "right_shoulder_outer" not in kp.get("keypointlabels", [])
            and "right_shoulder_inner" not in kp.get("keypointlabels", [])
            and "left_shoulder_outer" not in kp.get("keypointlabels", [])
            and "left_shoulder_inner" not in kp.get("keypointlabels", [])
            and "right_back_knee_inner" not in kp.get("keypointlabels", [])
            and "right_back_knee_outer" not in kp.get("keypointlabels", [])
            and "left_back_knee_inner" not in kp.get("keypointlabels", [])
            and "left_back_knee_outer" not in kp.get("keypointlabels", [])
            and "neck_top" not in kp.get("keypointlabels", [])
            and "neck_bottom" not in kp.get("keypointlabels", [])
        ]
        
        # Update annotation keypoints
        annotation["kp"] = keypoints
    # Save updated JSON data to a file
    output_file = '/home/vip24_shared/mmpose/visible_keypoints/midpoint_consolidated.json'
    with open(output_file, 'w') as file:
        json.dump(data, file, indent=4)

    print(f"Updated data successfully saved to {output_file}")

def keypoint_average(keypoints, label):
    """Calculate the average (midpoint) of all keypoints with the same label."""
    if not keypoints:
        return None
    
    avg_x = sum(kp["x"] for kp in keypoints) / len(keypoints)
    avg_y = sum(kp["y"] for kp in keypoints) / len(keypoints)
    avg_width = sum(kp.get("width", 0) for kp in keypoints) / len(keypoints)  # Optional
    return {
        "x": avg_x,
        "y": avg_y,
        "width": avg_width,  # Optional
        "keypointlabels": [label],
        "original_width": keypoints[0]["original_width"],
        "original_height": keypoints[0]["original_height"]
    }

def labeler_consolidation():
    # Load JSON file
    with open('/home/vip24_shared/mmpose/visible_keypoints/midpoint_consolidated.json', 'r') as file:
        data = json.load(file)

    # Group keypoints by image
    keypoints_by_image = {}

    for annotation in data:
        image = annotation["image"]
        if image not in keypoints_by_image:
            keypoints_by_image[image] =  {
            "id": annotation["id"],  # Preserve original ID
            "keypoints": []
        }
        keypoints_by_image[image]["keypoints"].extend(annotation["kp"])


    # Process keypoints for each image
    consolidated_annotations = []

    for image, content in keypoints_by_image.items():
        original_id = content["id"]  # Retrieve the original ID
        keypoints = content["keypoints"]
        
        # Group keypoints by label
        grouped_keypoints = {}
        for kp in keypoints:
            for label in kp.get("keypointlabels", []):
                if label not in grouped_keypoints:
                    grouped_keypoints[label] = []
                grouped_keypoints[label].append(kp)


        # Calculate consolidated keypoints for each label
        consolidated_keypoints = []
        for label, label_keypoints in grouped_keypoints.items():
            consolidated_keypoint = keypoint_average(label_keypoints, label)
            if consolidated_keypoint:
                consolidated_keypoints.append(consolidated_keypoint)
        
        consolidated_annotations.append({
            "image": image,
            "id": original_id,  # Use the preserved ID
            "kp": consolidated_keypoints
        })


    # Save updated JSON data to a file
    output_file = '/home/vip24_shared/mmpose/visible_keypoints/consolidated_annotations.json'
    with open(output_file, 'w') as file:
        json.dump(consolidated_annotations, file, indent=4)

    print(f"Updated data successfully saved to {output_file}")

def label_to_coco ():
    coco_format = {"info": {}, "licenses":[], "images": [], "annotations": [], "categories": []}
    keypoint_labels = ["left_eye", "right_eye", "nose", "neck", "root_of_tail", "left_shoulder", "left_elbow",
    "left_front_paw", "right_shoulder", "right_elbow", "right_front_paw", "left_hip", "left_knee", "left_back_paw",
    "right_hip", "right_knee", "right_back_paw"]
    coco_format["info"] = {
        "description": "AP-10k",
        "url": "https://github.com/AlexTheBad/AP-10K",
        "version": "1.0",
        "year": 2021,
        "contributor": "AP-10k Team",
        "date_created": "2021/07/01"
        }
    coco_format["licenses"].append([
        {
            "id": 1,
            "name": "The MIT License",
            "url": "https://www.mit.edu/~amini/LICENSE.md"
        }
        ])
    category = {
            "id": 1,
            "name": "antelope",
            "supercategory": "Bovidae",
            "keypoints": keypoint_labels,
            "skeleton": [[1,2],[2,3],[3,4],[4,5],[5,6],[6,7],[7,8],[8,9],[9,10],[10,11],[11,12]]
        
    }
    coco_format["categories"].append(category)
    
    annotation_id = 1
    # Load JSON file
    with open('/home/vip24_shared/mmpose/visible_keypoints/consolidated_annotations.json', 'r') as file:
        data = json.load(file)
    for item in data:
        image_id = int(item["image"].split("/")[-1].split(".")[0].lstrip("0"))
        image = {
            "license": 1,
            "id": image_id,
            "file_name": item["image"].split("/")[-1],
            "width": item["kp"][0]["original_width"],
            "height": item["kp"][0]["original_height"],
            "background": 0
        }
        coco_format["images"].append(image)
        
        keypoints = [0] * (len(keypoint_labels) * 3)
        
        for kp in item["kp"]:
            label = kp["keypointlabels"][0]
            if label in keypoint_labels:
                idx = keypoint_labels.index(label) * 3
                x = round((kp["x"] / 100) * item["kp"][0]["original_width"])
                y = round((kp["y"] / 100) * item["kp"][0]["original_height"])

                keypoints[idx:idx+3] = [x, y, 2]  # 2 means visible
        
        annotation = {
            "id": item["id"],
            "image_id": image_id,
            "category_id": 1,
            "bbox": [],
            "area": 0,
            "iscrowd": 0,
            "num_keypoints": len(item["kp"]),
            "keypoints": keypoints,
        }
        coco_format["annotations"].append(annotation)
    
    def merge_bbox_area(coco_data):
        # Load the additional annotation file
        bbox_files = ["/home/vip24_shared/mmpose/data/ap10k/annotations/ap10k-test-split1.json",
                    "/home/vip24_shared/mmpose/data/ap10k/annotations/ap10k-train-split1.json",
                    "/home/vip24_shared/mmpose/data/ap10k/annotations/ap10k-val-split1.json"]

        bbox_lookup = {}

        # Load bbox and area data from multiple files
        for bbox_file in bbox_files:
            with open(bbox_file, "r") as f:
                bbox_area_data = json.load(f)
                for ann in bbox_area_data["annotations"]:
                    if ann["image_id"] not in bbox_lookup:
                        bbox_lookup[ann["image_id"]] = {"bbox": ann["bbox"], "area": ann["area"]}

        # Update annotations with bbox and area
        for ann in coco_data["annotations"]:
            if ann["image_id"] in bbox_lookup:
                ann["bbox"] = bbox_lookup[ann["image_id"]]["bbox"]
                ann["area"] = bbox_lookup[ann["image_id"]]["area"]

        return coco_data
    coco_format = merge_bbox_area(coco_format)
     # Save updated JSON data to a file
    output_file = '/home/vip24_shared/mmpose/visible_keypoints/visible_annotations_total.json'
    with open(output_file, 'w') as file:
        json.dump(coco_format, file, indent=4)

    print(f"Updated data successfully saved to {output_file}")

def split_coco_annotations(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    with open('/home/vip24_shared/mmpose/visible_keypoints/visible_annotations_total.json', 'r') as file:
        data = json.load(file)
    images = data["images"]
    annotations = data["annotations"]
    
    random.shuffle(images)  # Shuffle images for randomness
    num_train = int(len(images) * train_ratio)
    num_val = int(len(images) * val_ratio)
    
    train_images = images[:num_train]
    val_images = images[num_train:num_train + num_val]
    test_images = images[num_train + num_val:]
    output_file = '/home/vip24_shared/mmpose/visible_keypoints/test_images.json'
    with open(output_file, 'w') as file:
        json.dump(test_images, file, indent=4)
    output_file = '/home/vip24_shared/mmpose/visible_keypoints/train_images.json'
    with open(output_file, 'w') as file:
        json.dump(train_images, file, indent=4)
    output_file = '/home/vip24_shared/mmpose/visible_keypoints/val_images.json'
    with open(output_file, 'w') as file:
        json.dump(val_images, file, indent=4)
    def get_annotations(image_subset):
        image_ids = {img["id"] for img in image_subset}
        return [ann for ann in annotations if ann["image_id"] in image_ids]
    
    datasets = {
        "train": {"info": data["info"], "licenses": data["licenses"], "images": train_images, "annotations": get_annotations(train_images), "categories": data["categories"]},
        "val": {"info": data["info"], "licenses": data["licenses"],"images": val_images, "annotations": get_annotations(val_images), "categories": data["categories"]},
        "test": {"info": data["info"], "licenses": data["licenses"],"images": test_images, "annotations": get_annotations(test_images), "categories": data["categories"]}
    }
    
    for split, data in datasets.items():
        with open(f"/home/vip24_shared/mmpose/visible_keypoints/{split}_visible_annotations.json", "w") as f:
            json.dump(data, f, indent=2)
def create_ap10k_antelope_annotations():
    all_annotations = {"images": [], "annotations": [], "categories": None}
    annotation_files = ["/home/vip24_shared/mmpose/data/ap10k/annotations/ap10k-test-split1.json",
                    "/home/vip24_shared/mmpose/data/ap10k/annotations/ap10k-train-split1.json",
                    "/home/vip24_shared/mmpose/data/ap10k/annotations/ap10k-val-split1.json"]
    for file in annotation_files:
        with open(file, "r") as f:
            data = json.load(f)
            all_annotations["images"].extend(data["images"])
            all_annotations["annotations"].extend(data["annotations"])
            if all_annotations["categories"] is None:
                all_annotations["categories"] = data["categories"]
    image_files = {"test": '/home/vip24_shared/mmpose/visible_keypoints/test_images.json',
                "train": '/home/vip24_shared/mmpose/visible_keypoints/train_images.json',
                "val": '/home/vip24_shared/mmpose/visible_keypoints/val_images.json'}
    # Load image lists for train, val, test
    datasets = {}
    for split, file in image_files.items():
        with open(file, "r") as f:
            datasets[split] = json.load(f)

    # Convert image lists to sets for quick lookup
    image_id_sets = {split: {img["id"] for img in datasets[split]} for split in datasets}

    # Filter annotations for each split
    for split in datasets:
        filtered_data = {
            "info": data["info"],
            "licenses": data["licenses"],
            "images": datasets[split],  # Keep only matching images
            "annotations": [ann for ann in all_annotations["annotations"] if ann["image_id"] in image_id_sets[split]],
            "categories": all_annotations["categories"]  # Keep category info
        }
        output_files = {
            "train": "/home/vip24_shared/mmpose/visible_keypoints/train_ap10k_annotations.json",
            "val": "/home/vip24_shared/mmpose/visible_keypoints/val_ap10k_annotations.json",
            "test": "/home/vip24_shared/mmpose/visible_keypoints/test_ap10k_annotations.json"
        }
        # Save the filtered annotations
        with open(output_files[split], "w") as f:
            json.dump(filtered_data, f, indent=2)

        print(f"Saved {split} annotations to {output_files[split]}")

#midpoint_consolidation()
#labeler_consolidation()
#label_to_coco()
#split_coco_annotations()
create_ap10k_antelope_annotations()
