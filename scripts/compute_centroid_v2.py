import os
import json
import numpy as np
import pandas as pd
from collections import defaultdict

def compute_keypoint_variations(keypoints):
    """
    Calculate coefficient of variation (CV) for keypoint distances from the centroid.

    Parameters:
    -----------
    keypoints : list
        Flattened keypoints array with [x, y, visibility] for each keypoint

    Returns:
    --------
    numpy.ndarray
        Array of coefficient of variation for each keypoint,
        or -np.inf for invisible keypoints
    """
    keypoints = np.array(keypoints).reshape(-1, 3)
    visible = keypoints[:, 2] > 0  # Mask for visible keypoints

    if not np.any(visible):
        return np.full(len(keypoints), -np.inf)  # No visible keypoints

    # Calculate centroid using only visible keypoints
    centroid = np.mean(keypoints[visible, :2], axis=0)

    # Calculate distances from centroid
    distances = np.full(len(keypoints), -np.inf)  # Default to -inf for invisible keypoints
    distances[visible] = np.linalg.norm(keypoints[visible, :2] - centroid, axis=1)

    # If all distances are zero, return zeros to avoid division by zero
    if np.all(distances[visible] == 0):
        variations = np.zeros(len(keypoints))
        variations[~visible] = -np.inf
        return variations

    # Calculate coefficient of variation (CV)
    # CV = standard deviation / mean (slightly not a CV its a variation of it)
    visible_distances = distances[visible]
    if np.mean(visible_distances) == 0:
        variations = np.zeros(len(keypoints))
    else:
        variations = np.zeros(len(keypoints))

        variations[visible] = visible_distances / np.mean(visible_distances)

    # Set non-visible keypoints to -inf
    variations[~visible] = -np.inf

    return variations

def process_annotations(annotation_files, base_path, output_dir):
    """
    Extract keypoint variations for each species and save as CSV files.

    Parameters:
    -----------
    annotation_files : list
        List of annotation JSON filenames
    base_path : str
        Base directory path
    output_dir : str
        Directory to save output CSV files

    Returns:
    --------
    dict
        Dictionary of species data with variations
    """
    os.makedirs(output_dir, exist_ok=True)

    species_data = defaultdict(list)
    exclude_images = set()

    print("Scanning for multi-annotation images...")
    for filename in annotation_files:
        filepath = os.path.join(base_path, "data", "ap10k", "annotations", filename)
        with open(filepath, 'r') as file:
            data = json.load(file)

        counts = defaultdict(int)
        for ann in data['annotations']:
            counts[ann['image_id']] += 1
        exclude_images.update({img_id for img_id, c in counts.items() if c > 1})

    print("Analyzing keypoint variations...")
    for filename in annotation_files:
        filepath = os.path.join(base_path, "data", "ap10k", "annotations", filename)
        with open(filepath, 'r') as file:
            data = json.load(file)

        category_map = {c['id']: c['name'] for c in data['categories']}

        for ann in data['annotations']:
            if ann['image_id'] in exclude_images or ann.get('iscrowd', 0) == 1:
                continue

            species = category_map[ann['category_id']]
            variations = compute_keypoint_variations(ann['keypoints'])
            species_data[species].append({'image_id': ann['image_id'], 'variations': variations})

    print("\nSaving results...")
    for species, records in species_data.items():
        if not records:
            continue

        df = pd.DataFrame([
            {**{'image_id': r['image_id']},
             **{f'keypoint_{i+1}_variation': d for i, d in enumerate(r['variations'])}}
            for r in records
        ])
        df.to_csv(os.path.join(output_dir, f"{species}_keypoint_variations.csv"), index=False)
        print(f"{species}: {len(df)} samples saved.")

    return species_data

def main():
    annotation_filenames = ["ap10k-train-split1.json", "ap10k-val-split1.json", "ap10k-test-split1.json"]
    base_path = os.path.abspath(os.path.dirname(__file__))
    output_dir = os.path.join(base_path, "centroid_variation_work_v2")

    species_data = process_annotations(annotation_filenames, base_path, output_dir)

    print(f"\nCSVs for {len(species_data)} species saved in {output_dir}")

    # Show example data
    if species_data:
        sample_species = next(iter(species_data))
        df = pd.read_csv(os.path.join(output_dir, f"{sample_species}_keypoint_variations.csv"))
        print(f"\nSample data for {sample_species}:")
        print(df.head())

if __name__ == "__main__":
    main()