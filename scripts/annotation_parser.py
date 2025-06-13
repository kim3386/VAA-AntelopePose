import os
import json
import shutil

# INPUTS
target_species = ["antelope", "horse"]
output_filename = 'test_filtered_annotations.json'  # Output filename in filtered_ap10k/annotations folder
# ------------------------------------------------------------------------

# Paths setup
annotation_filename = "ap10k-train-split1.json"
curr_dir_path = os.path.abspath(os.path.dirname(__file__))
annotation_filepath = os.path.join(curr_dir_path, "data", "ap10k", "annotations", annotation_filename)
output_folder_path = os.path.join(curr_dir_path, "filtered_ap10k", "data")
input_folder_path = os.path.join(curr_dir_path, "data", "ap10k", "data")

# Load the original annotation data
with open(annotation_filepath, 'r') as file:
    data = json.load(file)

info = data.get("info", {})
licenses = data.get("licenses", [])

# Find category IDs for the target species
target_category_ids = [
    category['id'] for category in data['categories'] if category['name'] in target_species
]

# Filter annotations and image IDs associated with the target species
filtered_annotations = [
    annotation for annotation in data['annotations'] if annotation['category_id'] in target_category_ids
]
filtered_image_ids = {annotation['image_id'] for annotation in filtered_annotations}

# Filter the images based on the filtered image IDs
filtered_images = [
    image for image in data['images'] if image['id'] in filtered_image_ids
]

# Filter the categories to only include the relevant species
filtered_categories = [
    category for category in data['categories'] if category['id'] in target_category_ids
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
output_filepath = os.path.join(curr_dir_path, "filtered_ap10k", "annotations", output_filename)
os.makedirs(os.path.dirname(output_filepath), exist_ok=True)  # Ensure the directory exists
with open(output_filepath, 'w') as output_file:
    json.dump(output_data, output_file)

print(f"Outputted filtered annotations to {output_filepath}")
