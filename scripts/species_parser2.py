import os
import json
import shutil
import numpy as np
from collections import defaultdict

# INPUTS
IMAGES_PER_SPECIES = 50
# ------------------------------------------------------------------------

def is_sideways_pose(keypoints):
    """
    Check if the animal is in a sideways pose by analyzing keypoint positions.
    Returns True if:
    1. Either nose or eyes are the rightmost visible points
    2. Either root of tail or hips are the leftmost visible points
    """
    keypoints = np.array(keypoints).reshape(-1, 3)
    visible_mask = keypoints[:, 2] > 0
    visible_points = keypoints[visible_mask]

    if len(visible_points) < 4:
        return False

    x_coords = visible_points[:, 0]
    y_coords = keypoints[:, 1]

    point_indices = np.where(visible_mask)[0]

    rightmost_idx = point_indices[np.argmax(x_coords)]
    leftmost_idx = point_indices[np.argmin(x_coords)]

    valid_right = rightmost_idx in [0, 1, 2]  # Left Eye, Right Eye, or Nose [0, 1, 2]
    valid_left = leftmost_idx in [4, 11, 14]  # Root of Tail, Left Hip, or Right Hip [4, 11, 14]

    inverse_valid_right = rightmost_idx in [4, 11, 14]
    inverse_valid_left = leftmost_idx in [0, 1, 2]

    if 4 in point_indices and 5 in point_indices:
        y4, y5 = y_coords[4], y_coords[5]
        if min(y4, y5) / max(y4, y5) < 0.8: #mess around with this a little
            return False


    '''if 4 not in point_indices or 5 not in point_indices:
        return False''' # Filters too much

    return (valid_right and valid_left) or (inverse_valid_left and inverse_valid_right)

# Paths setup
annotation_filenames = ["ap10k-train-split1.json", "ap10k-val-split1.json", "ap10k-test-split1.json"]
curr_dir_path = os.path.abspath(os.path.dirname(__file__))
output_base_path = os.path.join(curr_dir_path, "same_pose_ap10k")
input_folder_path = os.path.join(curr_dir_path, "data", "ap10k", "data")

# Initialize containers
species_counts = defaultdict(int)
selected_images = defaultdict(set)  # Dictionary to store image IDs by species
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
            selected_images[species_name].add(annotation['image_id'])
            species_counts[species_name] += 1

# Copy selected images to species-specific folders
total_copied = 0
print("\nCopying images to species-specific folders...")

for species_name, image_ids in selected_images.items():
    # Create species-specific subfolder
    species_folder = os.path.join(output_base_path, species_name)
    os.makedirs(species_folder, exist_ok=True)

    # Copy images for this species
    for filename in os.listdir(input_folder_path):
        file_path = os.path.join(input_folder_path, filename)
        if os.path.isfile(file_path):
            file_without_extension = filename.split('.')[0]
            try:
                image_id = int(file_without_extension)
                if image_id in image_ids:
                    destination_file_path = os.path.join(species_folder, filename)
                    shutil.copy(file_path, destination_file_path)
                    total_copied += 1
                    print(f"Copied {filename} to {species_name} folder ({total_copied} total images copied)")
            except ValueError:
                continue

# Print statistics
print("\nProcessing complete!")
print(f"Skipped {len(images_with_multiple_annotations)} images with multiple animals")
print("\nImages selected per species:")
for species, count in sorted(species_counts.items()):
    print(f"{species}: {count} images")
print(f"\nTotal images copied: {total_copied}")
print(f"Images organized into {len(selected_images)} species folders")
