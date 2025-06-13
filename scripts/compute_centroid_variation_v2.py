import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

def load_species_keypoint_variations(directory):
    """
    Load average keypoint variations for each species from CSV files.

    Parameters:
    -----------
    directory : str
        Directory containing species variation CSV files

    Returns:
    --------
    dict
        Dictionary with species names as keys and average keypoint variations as values
    """
    species_variations = {}

    for filename in os.listdir(directory):
        if filename.endswith('_keypoint_variations.csv'):
            species = filename.replace('_keypoint_variations.csv', '')
            filepath = os.path.join(directory, filename)

            # Read the CSV
            df = pd.read_csv(filepath)

            # Find keypoint variation columns (excluding 'image_id')
            variation_columns = [col for col in df.columns if col.startswith('keypoint_')]

            # Compute average variations, ignoring -inf values
            avg_variations = df[variation_columns].replace(-np.inf, np.nan).mean()

            species_variations[species] = avg_variations.values
            avg_variations = avg_variations.fillna(0)

            # Ensure species has valid data before adding
            if not avg_variations.isnull().all():
                species_variations[species] = avg_variations.values

    return species_variations

def compute_species_similarity(species_variations):
    """
    Compute cosine similarity between species based on keypoint variations.

    Parameters:
    -----------
    species_variations : dict
        Dictionary with species names and their average keypoint variations

    Returns:
    --------
    pd.DataFrame
        Similarity matrix with species names as index and columns
    """
    # Convert variations to a 2D numpy array
    species_names = list(species_variations.keys())
    variations_array = np.array(list(species_variations.values()))

    # Compute cosine similarity
    similarity_matrix = cosine_similarity(variations_array)

    # Create a DataFrame with species names
    similarity_df = pd.DataFrame(
        similarity_matrix,
        index=species_names,
        columns=species_names
    )

    return similarity_df

def find_closest_to_antelope(similarity_df, top_n=10):
    """
    Find species most similar to antelope based on cosine similarity.

    Parameters:
    -----------
    similarity_df : pd.DataFrame
        Cosine similarity matrix
    top_n : int, optional
        Number of top similar species to return

    Returns:
    --------
    pd.Series
        Top N species most similar to antelope, sorted by similarity
    """
    antelope_similarities = similarity_df.loc['antelope'].sort_values(ascending=False)

    # Exclude self-similarity bc it is 1.0 (0.999999999 is to ignore antelope)
    antelope_similarities = antelope_similarities[(antelope_similarities < 1.0) & (antelope_similarities != 0.9999999999999999)]

    return antelope_similarities.head(top_n), antelope_similarities.tail(top_n)

def main():
    base_path = os.path.abspath(os.path.dirname(__file__))
    variation_dir = os.path.join(base_path, "centroid_variation_work_v2") #change between centroid_variation_no_head and centroid_variation_work_v2
    output_dir = os.path.join(base_path, "centroid_variation_work_v2/centroid_similarity_analysis")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load species keypoint variations
    species_variations = load_species_keypoint_variations(variation_dir)

    # Compute similarity matrix
    similarity_df = compute_species_similarity(species_variations)
    print("\nFull Similarity Matrix:")
    print(similarity_df)

    # Find species most similar to antelope
    closest_to_antelope = find_closest_to_antelope(similarity_df)

    if closest_to_antelope is not None:
        print("\nTop 10 Species Most Similar to Antelope:")
        print(closest_to_antelope[0])

        print("\nBottom 10 Species Most Similar to Antelope:")
        print(closest_to_antelope[1])
        # Visualize with a bar plot
        plt.figure(figsize=(12, 6))
        closest_to_antelope[0].plot(kind='bar')
        plt.title('Species Similarity to Antelope')
        plt.xlabel('Species')
        plt.ylabel('Cosine Similarity')
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0.90, 1.0)  # Zoom in on a specific range

        plt.tight_layout()

        # Save the plot
        plot_path = os.path.join(output_dir, 'antelope_similarity_plot.png')
        plt.savefig(plot_path)
        print(f"\nPlot saved to {plot_path}")

        # Export closest species to CSV
        csv_path = os.path.join(output_dir, 'antelope_closest_species.csv')
        closest_to_antelope[0].to_csv(csv_path)
        print(f"Closest species data saved to {csv_path}")

if __name__ == "__main__":
    main()
