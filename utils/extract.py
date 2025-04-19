import os
import cv2
import json
import numpy as np

def find_matching_foundations(lab_values_file, oily_db='oily.csv', drytonormal_db='dry_to_normal.csv'):
    """
    Find the top match and a lighter shade for both oily foundations and dry-to-normal foundations.
    The function returns a dictionary with 4 foundation matches:
      - 'oily_best': best match from oily_db (lowest distance)
      - 'oily_lighter': first oily foundation with an L value higher than the best match
      - 'dry_best': best match from drytonormal_db (lowest distance)
      - 'dry_lighter': first dry-to-normal foundation with an L value higher than the best match
    Each match is a tuple: (foundation_name, distance, lab_values, image_path).
    If a lighter shade is not found, the best match is returned as the lighter option.
    
    :param lab_values_file: Path to the JSON file containing skin LAB values (with key 'lab_values')
    :param oily_db: Path to the CSV file for oily foundations (columns: Name, L, A, B)
    :param drytonormal_db: Path to the CSV file for dry-to-normal foundations (columns: Name, L, A, B)
    :return: Dictionary with the four foundation matches.
    """
    # Load the skin LAB values
    try:
        with open(lab_values_file, 'r') as f:
            skin_data = json.load(f)
        if 'lab_values' not in skin_data:
            print(f"Error: 'lab_values' key not found in {lab_values_file}")
            return {}
        skin_lab = skin_data['lab_values']
    except Exception as e:
        print(f"Error loading skin LAB values: {e}")
        return {}

    def process_db(csv_file):
        """
        Process a CSV file to compute Euclidean distances of foundation LAB values relative to skin_lab.
        Returns a tuple (best_match, lighter_match). Each match is a tuple: 
            (foundation_name, distance, lab_values, image_path)
        """
        foundations = []
        try:
            with open(csv_file, 'r') as f:
                # Skip header row
                next(f)
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) >= 4:  # Columns: Name, L, A, B
                        name = parts[0]
                        try:
                            lab_values = [float(parts[1]), float(parts[2]), float(parts[3])]
                        except ValueError:
                            continue
                        # Euclidean distance calculation
                        distance = np.sqrt(sum((s - l) ** 2 for s, l in zip(skin_lab, lab_values)))
                        foundations.append((name, distance, lab_values))
        except Exception as e:
            print(f"Error loading foundation database from {csv_file}: {e}")
            return None, None

        if not foundations:
            return None, None

        # Sort foundations by distance (ascending)
        foundations_sorted = sorted(foundations, key=lambda x: x[1])
        best_match = foundations_sorted[0]
        best_L = best_match[2][0]

        # Find the first foundation with a higher L value (lighter shade) than best_match
        lighter_match = None
        for candidate in foundations_sorted:
            if candidate[2][0] > best_L:
                lighter_match = candidate
                break
        # If not found, use best_match as lighter_match as well.
        if lighter_match is None:
            lighter_match = best_match

        # Append image_path to each match
        def append_image(match):
            name, distance, lab_values = match
            image_path = f"foundation-pictures/{name}/{name}.jpg"
            return (name, distance, lab_values, image_path)

        return append_image(best_match), append_image(lighter_match)

    oily_best, oily_lighter = process_db(oily_db)
    dry_best, dry_lighter = process_db(drytonormal_db)

    results = {
        'oily_best': oily_best,        # (name, distance, [L, A, B], image_path)
        'oily_lighter': oily_lighter,  # (name, distance, [L, A, B], image_path)
        'dry_best': dry_best,          # (name, distance, [L, A, B], image_path)
        'dry_lighter': dry_lighter     # (name, distance, [L, A, B], image_path)
    }
    return results


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
