import os
import json
import shutil
import random

# INPUTS
target_species = ["antelope"]
output_filename1 = 'test_annotations.json'
output_filename2 = 'finetune_annotations.json'  # Output filename in filtered_ap10k/annotations folder
# ------------------------------------------------------------------------

# Paths setup
annotation_filenames = ["ap10k-train-split1.json", "ap10k-val-split1.json", "ap10k-test-split1.json"]
curr_dir_path = os.path.abspath(os.path.dirname(__file__))
output_folder_path = os.path.join(curr_dir_path, "filtered_ap10k_test_antelopes", "data")
input_folder_path = os.path.join(curr_dir_path, "data", "ap10k", "data")

# Load the original annotation data

filtered_annotations = []
filtered_image_ids = []
filtered_images = []
filtered_categories = []  # Use set to avoid duplicate categories

for annotation_filename in annotation_filenames:
    annotation_filepath = os.path.join(curr_dir_path, "data", "ap10k", "annotations", annotation_filename)

    with open(annotation_filepath, 'r') as file:
        data = json.load(file)

    info = data.get("info", {})
    licenses = data.get("licenses", [])

    # Find category IDs for the target species
    target_category_ids = {
        category['id'] for category in data['categories'] if category['name'] in target_species
    }

    # Filter annotations and image IDs associated with the target species
    for annotation in data['annotations']:
        if annotation['category_id'] in target_category_ids:
            filtered_annotations.append(annotation)
            filtered_image_ids.append(annotation['image_id'])

    # Filter the images based on the filtered image IDs
    for image in data['images']:
        if image['id'] in filtered_image_ids:
            filtered_images.append(image)

    # Filter the categories to only include the relevant species
    for category in data['categories']:
        if category['id'] in target_category_ids:
            filtered_categories.append(category)

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

# Split annotations randomly into two sets
random.shuffle(filtered_annotations)
mid_index = len(filtered_annotations) // 2
annotation_train = filtered_annotations[:mid_index]
annotation_test = filtered_annotations[mid_index:]

# Separate images for each annotation subset to avoid duplication
image_ids_train = {annotation['image_id'] for annotation in annotation_train}
image_ids_test = {annotation['image_id'] for annotation in annotation_test}

filtered_images_train = [image for image in filtered_images if image['id'] in image_ids_train]
filtered_images_test = [image for image in filtered_images if image['id'] in image_ids_test]

# Prepare the output JSON with valid structure
output_data_train = {
    "info": info,
    "licenses": licenses,
    "images": filtered_images_train,
    "annotations": annotation_train,
    "categories": list(filtered_categories),  # Convert set to list for JSON serialization
}
output_data_test = {
    "info": info,
    "licenses": licenses,
    "images": filtered_images_test,
    "annotations": annotation_test,
    "categories": list(filtered_categories),  # Convert set to list for JSON serialization
}
# Write the filtered data to the output file
output_filepath1 = os.path.join(curr_dir_path, "filtered_ap10k_test_antelopes", "annotations", output_filename1)
output_filepath2 = os.path.join(curr_dir_path, "filtered_ap10k_test_antelopes", "annotations", output_filename2)
os.makedirs(os.path.dirname(output_filepath1), exist_ok=True)  # Ensure the directory exists
with open(output_filepath1, 'w') as output_file:
    json.dump(output_data_train, output_file)

print(f"Outputted filtered annotations to {output_filepath1}")

with open(output_filepath2, 'w') as output_file:
    json.dump(output_data_test, output_file)

print(f"Outputted filtered annotations to {output_filepath2}")
