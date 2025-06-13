import os
import json
import shutil
import random

## Annotation parser to filter out antelopes and reduce size of dataset (1169 train, 140 val)

# INPUTS
exclude_species = ["antelope","argali sheep", "bison", "buffalo", "cow", "sheep", "deer", "moose"]
output_train_filename = 'train_annotations.json'
output_val_filename = 'val_annotations.json'
num_train_annotations = 1169
num_val_annotations = 140

# Paths setup
annotation_filename = "ap10k-val-split1.json"
curr_dir_path = os.path.abspath(os.path.dirname(__file__))
annotation_filepath = os.path.join(curr_dir_path, "data", "ap10k", "annotations", annotation_filename)
output_folder_path = os.path.join(curr_dir_path, "filtered_ap10k_no_antelope_bovidae_cervidae", "data")
input_folder_path = os.path.join(curr_dir_path, "data", "ap10k", "data")

# Load the original annotation data
with open(annotation_filepath, 'r') as file:
    data = json.load(file)

info = data.get("info", {})
licenses = data.get("licenses", [])

# Find category IDs for the species to exclude
exclude_category_ids = [
    category['id'] for category in data['categories'] if category['name'] in exclude_species
]

# Filter annotations and image IDs not associated with the excluded species
filtered_annotations = [
    annotation for annotation in data['annotations'] if annotation['category_id'] not in exclude_category_ids
]

# Count annotations per species
species_annotation_count = {}
for annotation in filtered_annotations:
    category_id = annotation['category_id']
    if category_id not in species_annotation_count:
        species_annotation_count[category_id] = 0
    species_annotation_count[category_id] += 1

# Calculate the proportion of annotations to keep for each species
total_annotations = sum(species_annotation_count.values())
train_proportion = num_train_annotations / total_annotations
val_proportion = num_val_annotations / total_annotations

# Sample annotations based on the calculated proportions
train_annotations = []
val_annotations = []
for category_id, count in species_annotation_count.items():
    category_annotations = [annotation for annotation in filtered_annotations if annotation['category_id'] == category_id]
    random.shuffle(category_annotations)
    num_train = int(count * train_proportion)
    num_val = int(count * val_proportion)
    train_annotations.extend(category_annotations[:num_train])
    val_annotations.extend(category_annotations[num_train:num_train + num_val])

# Filter the images based on the filtered image IDs
train_image_ids = {annotation['image_id'] for annotation in train_annotations}
val_image_ids = {annotation['image_id'] for annotation in val_annotations}

train_images = [image for image in data['images'] if image['id'] in train_image_ids]
val_images = [image for image in data['images'] if image['id'] in val_image_ids]

# Filter the categories to exclude the relevant species
filtered_categories = [
    category for category in data['categories'] if category['id'] not in exclude_category_ids
]

# Copy relevant image files to the output folder
os.makedirs(output_folder_path, exist_ok=True)  # Create output folder if it doesn't exist
for filename in os.listdir(input_folder_path):
    file_path = os.path.join(input_folder_path, filename)
    if os.path.isfile(file_path):
        file_without_extension = filename.split('.')[0]
        number = int(file_without_extension)
        if number in train_image_ids or number in val_image_ids:
            print(f"Processing file: {filename}")
            destination_file_path = os.path.join(output_folder_path, filename)
            shutil.copy(file_path, destination_file_path)

# Prepare the output JSON with valid structure for training and validation sets
train_output_data = {
    "info": info,
    "licenses": licenses,
    "images": train_images,
    "annotations": train_annotations,
    "categories": filtered_categories,
}

val_output_data = {
    "info": info,
    "licenses": licenses,
    "images": val_images,
    "annotations": val_annotations,
    "categories": filtered_categories,
}

# Write the filtered data to the output files
train_output_filepath = os.path.join(curr_dir_path, "filtered_ap10k_no_antelope_bovidae_cervidae", "annotations", output_train_filename)
val_output_filepath = os.path.join(curr_dir_path, "filtered_ap10k_no_antelope_bovidae_cervidae", "annotations", output_val_filename)
os.makedirs(os.path.dirname(train_output_filepath), exist_ok=True)  # Ensure the directory exists
os.makedirs(os.path.dirname(val_output_filepath), exist_ok=True)  # Ensure the directory exists

with open(train_output_filepath, 'w') as train_output_file:
    json.dump(train_output_data, train_output_file)

with open(val_output_filepath, 'w') as val_output_file:
    json.dump(val_output_data, val_output_file)

print(f"Outputted filtered training annotations to {train_output_filepath}")
print(f"Outputted filtered validation annotations to {val_output_filepath}")
