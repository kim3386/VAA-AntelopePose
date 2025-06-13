import os
import json
import shutil
import random
import csv
from math import sqrt
import math

# INPUTS
#target_species = "cat" #change for different species (change output file for each)

# Iterates through all side-view species and outputs limb ratios to csv within /limb_ratios
species_list = os.listdir("/home/vip24_shared/mmpose/species_similarity_organized_ap10k")
#species_list = ["brown bear"]
for target_species in species_list:
    output_filename = target_species + '_limb.csv'
    # ------------------------------------------------------------------------

    # Paths setup
    annotation_filenames = ["ap10k-train-split1.json", "ap10k-val-split1.json", "ap10k-test-split1.json"]
    curr_dir_path = os.path.abspath(os.path.dirname(__file__))
    output_folder_path = os.path.join(curr_dir_path, "limb_ratios")
    input_folder_path = os.path.join(curr_dir_path, "species_similarity_organized_ap10k", target_species)

    # For each image, find the annotation file and the index in "images"
    image_index = {}
    for filename in os.listdir(input_folder_path):
        image_index[filename] = -1

    for annotation_filename in annotation_filenames:
        annotation_filepath = os.path.join(curr_dir_path, "data", "ap10k", "annotations", annotation_filename)

        with open(annotation_filepath, 'r') as file:
            data = json.load(file)
        for index, image in enumerate(data['images']):
            if image['file_name'] in image_index.keys():
                image_index[image['file_name']] = (annotation_filename, index)

    # Extract annotations, calculate limb lengths, and regularize
    for annotation_filename, index in image_index.values():
        annotation_filepath = os.path.join(curr_dir_path, "data", "ap10k", "annotations", annotation_filename)
        with open(annotation_filepath, 'r') as file:
            data = json.load(file)
        #Iterate over each image entry in the annotation file
        for image_entry in data['images']:
            image_name = image_entry['file_name']
            if image_name in image_index:
                annotation_index = -1
                for i, ann in enumerate(data['annotations']):
                    if ann['image_id']==image_entry['id']:
                        annotation_index=i
                        break
                image_index[image_name]=(annotation_filename,annotation_index)

    # Extract skeleton
    skeleton=[[1, 2], [1, 3], [2, 3], [3, 4], [4, 5], [4, 6], [6, 7], [7, 8], [4, 9], [9, 10], [10, 11], [5, 12], [12, 13], [13, 14], [5, 15], [15, 16], [16, 17]]
    """category_id=None
    for image_filename,(ann_file,ann_idx) in image_index.items():
        annotation_path =os.path.join(curr_dir_path,"data","ap10k","annotations",ann_file)
        with open(annotation_path,'r') as file:
            data=json.load(file)
            annotation_entry =data['annotations'][ann_idx]
            current_category_id = annotation_entry['category_id']
            category =None
            for categ in data['categories']:
                if categ['id'] ==current_category_id:
                    category=categ
                    break
            skeleton =category['skeleton']
            category_id =current_category_id
            break"""
    # Calculate ratios
    limb_data=[]
    cvs_headers=csv_headers = ["filename"] + [f"[{conn[0]},{conn[1]}]" for conn in skeleton]
    for filename,(ann_file,ann_idx) in image_index.items():
        annotation_path = os.path.join(curr_dir_path, "data", "ap10k", "annotations", ann_file)
        with open(annotation_path,"r") as f:
            data = json.load(f)
        annotation =data["annotations"][ann_idx]
        #if annotation["category_id"] != category_id:
        #    print("here")
        #    continue
        keypoints = annotation["keypoints"]
        bbox = annotation["bbox"]
        bbox_height = bbox[3]
        if bbox_height <= 0:
            continue
        entry = {"filename": filename}
        for conn in skeleton:
            kp1 = conn[0] - 1
            kp2 = conn[1] - 1
            x1 = keypoints[kp1*3]
            y1 = keypoints[kp1*3 + 1]
            v1 = keypoints[kp1*3 + 2]
            x2 = keypoints[kp2*3]
            y2 = keypoints[kp2*3 + 1]
            v2 = keypoints[kp2*3 + 2]
            if v1 >0 and v2>0:
                distance = sqrt((x2 - x1)**2 + (y2 - y1)**2)
                ratio = distance / bbox_height
                entry[f"[{conn[0]},{conn[1]}]"] = round(ratio, 4)
        limb_data.append(entry)
            
        # to see what the annotations look like
        #print(data['annotations'][index])
        #print(data['annotations'][index]['keypoints'])
        #print(data['annotations'][index]['bbox'])

        # To access the keypoints use data['annotations'][index]['keypoints'] (are in a specific order - check github readME/dataFormat)
        # To access the bounding box coordinates use data['annotations'][index]['bbox']
        # To create a "limb" take the euclidean distance between the coordinates of two points (ex: rightknee to right back paw)
        # To find all possible "limbs", can use the "skeleton" defined in the readME as well (ex: [1,2] refers to left eye and right eye connection)

        # Goal: find all the "limb" lengths for an image, divide by the bounding box height to regularize,
        # add it to a new dictionary within the limb_data list
        # Then, for the next image, do the same in a new dictionary
        # Note: This is just for antelopes so far, we need to do this for every species (can do so by changing the target_species at the top (all species we need are in species_similarity_organized_ap10k))
        
    #Saving to csv within limb_ratios/

    output_filepath = os.path.join(output_folder_path, output_filename)
    with open(output_filepath, 'w', newline='') as csvfile:

        #fieldnames = limb_ratios[0].keys()
        #writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer = csv.DictWriter(csvfile, fieldnames=csv_headers)
        writer.writeheader()
        writer.writerows(limb_data)


limb_ratios_dir = "limb_ratios"
species_files = [f for f in os.listdir(limb_ratios_dir)]
species_averages = {}

for file in species_files:
    species_name = file.replace('_limb.csv', '')
    with open(os.path.join(limb_ratios_dir, file), 'r') as f:
        reader = csv.DictReader(f)
        limb_sums = {}
        limb_counts = {}
        for row in reader:
            for limb in reader.fieldnames[1:]:
                value = row.get(limb)
                if value and float(value) > 0:
                    limb_sums[limb] = limb_sums.get(limb, 0.0) + float(value)
                    limb_counts[limb] = limb_counts.get(limb, 0) + 1
        avg_limbs = {limb: limb_sums[limb]/limb_counts[limb] if limb_counts.get(limb) else 0.0 
                     for limb in reader.fieldnames[1:]}
        species_averages[species_name] = avg_limbs
all_limbs = sorted({limb for avg in species_averages.values() for limb in avg.keys()})
species_vectors = {}
for species, avg_limbs in species_averages.items():
    vector = [avg_limbs.get(limb, 0.0) for limb in all_limbs]
    species_vectors[species] = vector
def cosine_similarity(vec_a, vec_b):
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    mag_a = math.sqrt(sum(a ** 2 for a in vec_a))
    mag_b = math.sqrt(sum(b ** 2 for b in vec_b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot_product / (mag_a * mag_b)

species_names = list(species_vectors.keys())
similarity_matrix = []
for species_a in species_names:
    row = []
    vec_a = species_vectors[species_a]
    for species_b in species_names:
        vec_b = species_vectors[species_b]
        similarity = cosine_similarity(vec_a, vec_b)
        row.append(round(similarity, 4)) 
    similarity_matrix.append(row)

output_csv = "species_cosine_similarity.csv"
with open(output_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Species'] + species_names)
    for i, species in enumerate(species_names):
        writer.writerow([species] + similarity_matrix[i])
