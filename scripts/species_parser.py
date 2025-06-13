import os
import json
import shutil
import numpy as np
from collections import defaultdict

# INPUTS
IMAGES_PER_SPECIES = 5
# ------------------------------------------------------------------------

def is_sideways_pose(keypoints):
    """
    Check if the animal is in a sideways pose by analyzing keypoint positions.
    Returns True if:
    1. Either nose or eyes are the rightmost visible points
    2. Either root of tail or hips are the leftmost visible points

    Keypoint indices:
    - Eyes: 0, 1 (Left Eye, Right Eye)
    - Nose: 2
    - Root of Tail: 4
    - Hips: 11, 14 (Left Hip, Right Hip)
    """
    # Convert keypoints to numpy array of [x, y, visibility]
    keypoints = np.array(keypoints).reshape(-1, 3)

    # Only consider visible keypoints (visibility > 0)
    visible_mask = keypoints[:, 2] > 0
    visible_points = keypoints[visible_mask]

    if len(visible_points) < 4:  # Need at least a few visible points
        return False

    # Get x-coordinates of visible points
    x_coords = visible_points[:, 0]
    point_indices = np.where(visible_mask)[0]

    # Find rightmost and leftmost point indices
    rightmost_idx = point_indices[np.argmax(x_coords)]
    leftmost_idx = point_indices[np.argmin(x_coords)]

    valid_right = rightmost_idx in [2]  # Left Eye, Right Eye, or Nose [0, 1, 2]
    valid_left = leftmost_idx in [4]  # Root of Tail, Left Hip, or Right Hip [4, 11, 14]

    left_eye_visible = keypoints[0, 2] > 0 # Ensures left eye isn't there in a side view

    return valid_right and valid_left and not left_eye_visible

# Paths setup
annotation_filenames = ["ap10k-train-split1.json", "ap10k-val-split1.json", "ap10k-test-split1.json"]
curr_dir_path = os.path.abspath(os.path.dirname(__file__))
output_folder_path = os.path.join(curr_dir_path, "species_similarity_ap10k", "data")
input_folder_path = os.path.join(curr_dir_path, "data", "ap10k", "data")

# Initialize containers
species_counts = defaultdict(int)
selected_image_ids = set()
images_with_multiple_annotations = set()

# First pass: identify images with multiple annotations
print("Identifying images with multiple animals...")
for annotation_filename in annotation_filenames:
    annotation_filepath = os.path.join(curr_dir_path, "data", "ap10k", "annotations", annotation_filename)

    with open(annotation_filepath, 'r') as file:
        data = json.load(file)

    # Count annotations per image
    image_annotation_count = defaultdict(int)
    for annotation in data['annotations']:
        image_annotation_count[annotation['image_id']] += 1

    # Add images with multiple annotations to exclusion set
    for image_id, count in image_annotation_count.items():
        if count > 1:
            images_with_multiple_annotations.add(image_id)

# Second pass: process annotations
print("Analyzing annotations for sideways poses...")
for annotation_filename in annotation_filenames:
    annotation_filepath = os.path.join(curr_dir_path, "data", "ap10k", "annotations", annotation_filename)

    with open(annotation_filepath, 'r') as file:
        data = json.load(file)

    # Create category ID to name mapping
    category_mapping = {category['id']: category for category in data['categories']}

    # Process annotations
    for annotation in data['annotations']:
        # Skip if image has multiple annotations or if iscrowd=1
        if (annotation['image_id'] in images_with_multiple_annotations or
            annotation.get('iscrowd', 0) == 1):
            continue

        category_id = annotation['category_id']
        species_name = category_mapping[category_id]['name']

        # Skip if we already have enough images for this species
        if species_counts[species_name] >= IMAGES_PER_SPECIES:
            continue

        # Check if pose is sideways
        if is_sideways_pose(annotation['keypoints']):
            selected_image_ids.add(annotation['image_id'])
            species_counts[species_name] += 1

# Copy selected images to output folder
os.makedirs(output_folder_path, exist_ok=True)
copied_count = 0
total_images = len(selected_image_ids)

print(f"\nCopying {total_images} selected images...")

for filename in os.listdir(input_folder_path):
    file_path = os.path.join(input_folder_path, filename)
    if os.path.isfile(file_path):
        file_without_extension = filename.split('.')[0]
        try:
            image_id = int(file_without_extension)
            if image_id in selected_image_ids:
                destination_file_path = os.path.join(output_folder_path, filename)
                shutil.copy(file_path, destination_file_path)
                copied_count += 1
                print(f"Copied {filename} ({copied_count}/{total_images})")
        except ValueError:
            continue

# Print statistics
print("\nProcessing complete!")
print(f"Skipped {len(images_with_multiple_annotations)} images with multiple animals")
print("\nImages selected per species:")
for species, count in sorted(species_counts.items()):
    print(f"{species}: {count} images")
print(f"\nTotal images copied: {copied_count}")