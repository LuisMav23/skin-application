import os
import cv2
import json
import numpy as np

def find_matching_foundations(lab_values_file, foundations_db='lab_values.csv', top_n=5):
    """
    Find the top N foundations that match the skin color in the given LAB values file.
    
    :param lab_values_file: Path to the JSON file containing LAB values of the skin
    :param foundations_db: Path to the CSV file containing foundation LAB values
    :param top_n: Number of top matches to return
    :return: List of tuples (foundation_name, distance) sorted by closest match first
    """
    # Load the skin LAB values
    try:
        with open(lab_values_file, 'r') as f:
            skin_data = json.load(f)
        
        if 'lab_values' not in skin_data:
            print(f"Error: 'lab_values' key not found in {lab_values_file}")
            return []
            
        skin_lab = skin_data['lab_values']
    except Exception as e:
        print(f"Error loading skin LAB values: {e}")
        return []
    
    # Load the foundation database from CSV
    try:
        foundations = []
        with open(foundations_db, 'r') as f:
            # Skip header row
            next(f)
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 4:  # Name, L, A, B
                    name = parts[0]
                    lab_values = [float(parts[1]), float(parts[2]), float(parts[3])]
                    foundations.append((name, lab_values))
    except Exception as e:
        print(f"Error loading foundation database: {e}")
        return []
    
    # Calculate Euclidean distance for each foundation
    distances = []
    for foundation_name, foundation_lab in foundations:
        # Calculate Euclidean distance between skin and foundation LAB values
        distance = np.sqrt(sum((a - b) ** 2 for a, b in zip(skin_lab, foundation_lab)))
        distances.append((foundation_name, distance))
    
    # Sort by distance (closest match first) and return top N
    return sorted(distances, key=lambda x: x[1])[:top_n]


def extract_lab_from_image(image_path):
    """
    Extract the average LAB values from a single image.
    
    :param image_path: Path to the image file
    :return: A list containing the average L, A, B values or None if image cannot be loaded
    """
    # Load the image in BGR format (OpenCV default)
    image = cv2.imread(image_path)
    
    if image is None:
        # If the image can't be loaded, return None
        print(f"Could not load image: {image_path}")
        return None
    
    # Convert from BGR to LAB
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # Reshape to a list of pixels and compute the mean
    mean_lab = np.mean(lab_image.reshape(-1, 3), axis=0)
    
    # Convert to a Python list for easier handling
    return mean_lab.tolist()

def extract_and_average_lab(root_dir, output_json='lab_values.json'):
    """
    Traverse the root directory, extract average LAB values for images in each folder,
    and store the results in a JSON file.

    :param root_dir: The path to the directory containing foundation folders.
    :param output_json: The name of the output JSON file to save results.
    """
    results = {}

    # Loop over each folder in the root directory
    for foundation_folder in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, foundation_folder)

        # Only process if it's a directory
        if os.path.isdir(folder_path):
            lab_values_per_folder = []

            # Read each image file in this subfolder
            for file_name in os.listdir(folder_path):
                # Check if the file is an image (you can adjust extensions as needed)
                if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_path = os.path.join(folder_path, file_name)
                    
                    lab_values = extract_lab_from_image(image_path)
                    if lab_values:
                        # Collect the mean LAB for this image
                        lab_values_per_folder.append(lab_values)

            # If we have collected any LAB values in this folder, average them
            if lab_values_per_folder:
                overall_mean_lab = np.mean(lab_values_per_folder, axis=0)

                # Convert to a Python list for JSON serialization
                overall_mean_lab_list = overall_mean_lab.tolist()

                # Store in our results dictionary
                results[foundation_folder] = overall_mean_lab_list

    # Write results to JSON file
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"LAB values have been written to {output_json}")

# Usage:
if __name__ == "__main__":
    # Replace with the path to your root directory
    root_directory = "Skin tone picture"
    extract_and_average_lab(root_directory, output_json="lab_values.json")
