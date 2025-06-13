import os
import json
import shutil
import random

# INPUTS: Target species and number of annotations in the train and val sets
#animal_categories = ["tiger", "wolf", "squirrel", "panda", "deer", "fox", "brown bear", "spider monkey", "rhino", "mole"] # Random
# animal_categories = ["deer", "moose", "zebra", "horse", "giraffe", "argali sheep", "sheep", "cow", "bison", "buffalo"] # Human choice
# animal_categories = ["deer", "giraffe", "cheetah", "moose", "argali sheep", "fox", "buffalo", "zebra", "rabbit", "leopard"] # Full COS
# animal_categories = ["deer", "giraffe", "moose", "bison", "rabbit", "cheetah", "argali sheep", "sheep", "zebra", "leopard"] # Full KNN
# animal_categories = ["deer", "giraffe", "fox", "sheep", "cheetah", "cow", "wolf", "rabbit", "dog", "bobcat"] # KP COS
# animal_categories = ["deer", "giraffe", "rabbit", "cheetah", "fox", "rhino", "wolf", "sheep", "moose", "zebra"] # KP KNN
#animal_categories = ["deer", "giraffe", "zebra", "bison", "argali sheep", "buffalo", "sheep", "cheetah", "cow", "fox"] # Patch COS
#animal_categories = ["deer", "giraffe", "bison", "rabbit", "fox", "argali sheep", "buffalo", "cheetah", "moose", "cow"] # Patch KNN
# animal_categories = ["deer", "giraffe", "rabbit", "bison", "buffalo", "cheetah", "moose", "rhino", "fox", "sheep"] # Fused Patch KNN
# animal_categories = ["deer", "giraffe", "sheep", "bison", "fox", "buffalo", "zebra", "cow", "wolf", "hippo"] # Fused KPPatch COS
# animal_categories = ["deer", "rabbit", "giraffe", "bison", "fox", "polar bear", "lion", "leopard", "rhino", "weasel"] # Fused KPPatch KNN
#animal_categories = ["deer", "giraffe", "rabbit", "bison", "buffalo", "cheetah", "moose", "rhino", "fox", "sheep"] # False SD KPPatch COS
#animal_categories = ["squirrel", "rabbit", "mouse", "moose", "skunk", "dog", "raccoon", "sheep", "otter", "deer"] # SD KPPatch KNN
#animal_categories = ["deer", "fox", "dog", "moose", "weasel", "mouse", "rat", "cow", "cat", "bobcat"] # SD COS
#animal_categories = ["deer", "giraffe", "zebra", "bison", "argali sheep", "buffalo", "sheep", "cheetah", "cow", "elephant"] # Fixed Patch COS
animal_categories = ["deer", "giraffe", "bison", "rabbit", "elephant", "argali sheep", "buffalo", "cheetah", "moose", "leopard"] # Fixed Patch KNN

max_allowed_annotations = 2942
# INPUTS: Output file names
train_output_filename = 'train_annotations.json'
val_output_filename = 'val_annotations.json'
data_root = os.path.join("/home/vip24_shared/mmpose/filtered_ap10k_ADD_NAME")

# Paths setup
annotation_filenames = ["ap10k-train-split1.json", "ap10k-val-split1.json", "ap10k-test-split1.json"]
curr_dir_path = os.path.abspath(os.path.dirname(__file__))
output_folder_path = os.path.join(data_root, "data")
input_folder_path = os.path.join(curr_dir_path, "data", "ap10k", "data")

filtered_annotations = []
filtered_image_ids = set()
filtered_images = []
filtered_categories = []

for annotation_filename in annotation_filenames:
    annotation_filepath = os.path.join(curr_dir_path, "data", "ap10k", "annotations", annotation_filename)

    with open(annotation_filepath, 'r') as file:
        data = json.load(file)

    info = data.get("info", {})
    licenses = data.get("licenses", [])

    target_category_ids = {
        category['id'] for category in data['categories'] if category['name'] != "antelope" and category['name'] in animal_categories
    }

    for annotation in data['annotations']:
        if annotation['category_id'] in target_category_ids:
            filtered_annotations.append(annotation)
            filtered_image_ids.add(annotation['image_id'])

    filtered_images.extend([image for image in data['images'] if image['id'] in filtered_image_ids])

    for category in data['categories']:
        if category['id'] in target_category_ids and category not in filtered_categories:
            filtered_categories.append(category)

# Ensure the total number of annotations does not exceed max_allowed_annotations
if len(filtered_annotations) > max_allowed_annotations:
    filtered_annotations = random.sample(filtered_annotations, max_allowed_annotations)

# Split dataset (80% train, 20% validation)
total_annotations = len(filtered_annotations)
val_size = max(1, int(0.2 * total_annotations)) if total_annotations >= 50 else 0
train_annotations = filtered_annotations[val_size:]
val_annotations = filtered_annotations[:val_size]

# Get the unique image IDs in each split
train_image_ids = {ann['image_id'] for ann in train_annotations}
val_image_ids = {ann['image_id'] for ann in val_annotations}

# Assign images to respective sets
train_images = [img for img in filtered_images if img['id'] in train_image_ids]
val_images = [img for img in filtered_images if img['id'] in val_image_ids]

# Copy relevant image files
os.makedirs(output_folder_path, exist_ok=True)
image_count_train = 0
image_count_val = 0
for filename in os.listdir(input_folder_path):
    file_path = os.path.join(input_folder_path, filename)
    if os.path.isfile(file_path):
        file_id = int(filename.split('.')[0])
        if file_id in train_image_ids:
            shutil.copy(file_path, os.path.join(output_folder_path, filename))
            image_count_train += 1
        elif file_id in val_image_ids:
            shutil.copy(file_path, os.path.join(output_folder_path, filename))
            image_count_val += 1

# Prepare JSON output
output_annotation_folder = os.path.join(data_root, "annotations")
os.makedirs(output_annotation_folder, exist_ok=True)

train_data = {"info": info, "licenses": licenses, "images": train_images, "annotations": train_annotations, "categories": list(filtered_categories)}
val_data = {"info": info, "licenses": licenses, "images": val_images, "annotations": val_annotations, "categories": list(filtered_categories)}

with open(os.path.join(output_annotation_folder, train_output_filename), 'w') as f:
    json.dump(train_data, f)
print(f"Saved train annotations to {train_output_filename}")

if val_size > 0:
    with open(os.path.join(output_annotation_folder, val_output_filename), 'w') as f:
        json.dump(val_data, f)
    print(f"Saved validation annotations to {val_output_filename}")

# Print dataset statistics
print(f"Total annotations in dataset: {total_annotations}")
print(f"Train set: {len(train_annotations)} annotations")
print(f"Validation set: {len(val_annotations)} annotations")

if total_annotations > 500:
    print("Validation set added due to dataset size.")