import os
import json
import shutil
import random

# INPUTS
target_species = ["argali sheep", "bison", "buffalo" , "cow", "sheep", "deer", "moose"]
#train_split = {"argali sheep":83 , "bison":202, "buffalo":157 , "cow":172, "sheep":268, "deer":155, "moose":132}
val_split = {"argali sheep":7 , "bison":24, "buffalo":18 , "cow":22, "sheep":28, "deer":21, "moose":19}
output_filename = 'val_annotations.json'  # Output filename in filtered_ap10k/annotations folder
# ------------------------------------------------------------------------

# Paths setup
annotation_filename = "ap10k-val-split1.json"
curr_dir_path = os.path.abspath(os.path.dirname(__file__))
annotation_filepath = os.path.join(curr_dir_path, "data", "ap10k", "annotations", annotation_filename)
output_folder_path = os.path.join(curr_dir_path, "filtered_ap10k_b+c_reduced", "data")
input_folder_path = os.path.join(curr_dir_path, "data", "ap10k", "data")

# Load the original annotation data
with open(annotation_filepath, 'r') as file:
    data = json.load(file)

info = data.get("info", {})
licenses = data.get("licenses", [])

# Find category IDs for the target species
target_category_dict = {
    category['id'] : category['name'] for category in data['categories'] if category['name'] in target_species
}
filtered_annotations = []
for category_id, category_name in target_category_dict.items():
    # Filter annotations for the current categories
    category_annotations = [annotation for annotation in data['annotations'] if annotation['category_id'] == category_id]
    num_samples = val_split[category_name]
    num_samples = min(num_samples, len(category_annotations))
    print(num_samples)
    filtered_annotations.extend(random.sample(category_annotations, num_samples))

#print(len(filtered_annotations))
filtered_image_ids = {annotation['image_id'] for annotation in filtered_annotations}

# Filter the images based on the filtered image IDs
filtered_images = [
    image for image in data['images'] if image['id'] in filtered_image_ids
]

# Filter the categories to only include the relevant species
filtered_categories = [
    category for category in data['categories'] if category['id'] in target_category_dict.keys()
]

# Copy relevant image files to the output folder
os.makedirs(output_folder_path, exist_ok=True)  # Create output folder if it doesn't exist
for filename in os.listdir(input_folder_path):
    file_path = os.path.join(input_folder_path, filename)
    if os.path.isfile(file_path):
        file_without_extension = filename.split('.')[0]
        number = int(file_without_extension)
        if number in filtered_image_ids:
            print(f"Processing file: {filename}")
            destination_file_path = os.path.join(output_folder_path, filename)
            shutil.copy(file_path, destination_file_path)

# Prepare the output JSON with valid structure
output_data = {
    "info" : info,
    "licenses" : licenses,
    "images": filtered_images,
    "annotations": filtered_annotations,
    "categories": filtered_categories,
}

# Write the filtered data to the output file
output_filepath = os.path.join(curr_dir_path, "filtered_ap10k_b+c_reduced", "annotations", output_filename)
os.makedirs(os.path.dirname(output_filepath), exist_ok=True)  # Ensure the directory exists
with open(output_filepath, 'w') as output_file:
    json.dump(output_data, output_file)

print(f"Outputted filtered annotations to {output_filepath}")
