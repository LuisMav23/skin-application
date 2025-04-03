import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
import os
import json
from datetime import datetime
from utils.extract import extract_lab_from_image, find_matching_foundations

# --- Initialize MediaPipe Face Mesh and Drawing utilities ---
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Left cheek landmark indices
left_cheek_indices = [120, 117, 187, 203]

def warp_cheek_to_square(cheek_points, image):
    if len(cheek_points) != 4:
        return None  # Ensure exactly 4 points

    # Define the destination square (100x100 pixels)
    target_square = np.array([[0, 0], [99, 0], [99, 99], [0, 99]], dtype=np.float32)

    # Convert cheek points to a float32 NumPy array
    cheek_points = np.array(cheek_points, dtype=np.float32)

    # Compute perspective transform matrix
    M = cv2.getPerspectiveTransform(cheek_points, target_square)

    # Apply transformation to get a squared cheek region
    warped_cheek = cv2.warpPerspective(image, M, (100, 100))
    
    return warped_cheek

def process_frame(image):
    h, w, _ = image.shape
    # Convert to RGB as required by MediaPipe
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)
    annotated_image = image.copy()
    cheek_region = None

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Draw face mesh landmarks for feedback
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=1)
            )
            # Extract points corresponding to left cheek indices
            points = [[int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)] 
                      for i in left_cheek_indices]

            if len(points) == 4:
                points = np.array(points)
                hull = cv2.convexHull(points)
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillConvexPoly(mask, hull, 255)

                # Apply mask to get the full cheek region from the frame
                cheek_full = cv2.bitwise_and(image, image, mask=mask)
                x, y, w_box, h_box = cv2.boundingRect(hull)
                cheek_region = cheek_full[y:y+h_box, x:x+w_box]

                if cheek_region is not None and cheek_region.size > 0:
                    # Warp cheek to a square
                    warped_cheek = warp_cheek_to_square(points, image)

                    if warped_cheek is not None:
                        cheek_region = warped_cheek  # Replace with squared cheek

    return annotated_image, cheek_region

def save_cheek_image(cheek_image):
    # Create a directory to save images if it doesn't exist
    save_dir = "saved_cheeks"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Generate a filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{save_dir}/cheek_{timestamp}.jpg"
    
    # Save the image
    if cheek_image is not None and cheek_image.size > 0:
        cv2.imwrite(filename, cheek_image)
        return filename
    else:
        return "No valid cheek image to save"

def save_lab_values(cheek_image):
    lab_values = extract_lab_from_image(cheek_image)
    # Create a directory to save lab values if it doesn't exist
    lab_dir = "lab_values"
    if not os.path.exists(lab_dir):
        os.makedirs(lab_dir)
    
    # Generate a filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    lab_filename = f"{lab_dir}/lab_{timestamp}.json"
    
    # Save the lab values to a JSON file
    if lab_values:
        with open(lab_filename, 'w') as f:
            json.dump({"lab_values": lab_values}, f, indent=4)
        print(f"LAB values saved to {lab_filename}")
    if lab_values:
        return lab_filename, lab_values
    else:
        return "No valid lab values to save"

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error opening camera")
        return
    
    # Variable to store the current cheek image
    current_cheek = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Unable to read from camera")
            break

        # Process the current frame
        annotated, cheek = process_frame(frame)
        current_cheek = cheek  # Store the current cheek image
        
        # Overlay prompt text for the user
        cv2.putText(annotated, "Position your face", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(annotated, "Press 'S' to save cheek image", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Live Face Mesh", annotated)

        # Show the extracted and transformed cheek region (if available)
        if cheek is not None:
            cv2.imshow("Extracted Cheek (Warped to Square)", cheek)
        else:
            cv2.imshow("Extracted Cheek", np.zeros((100, 100, 3), dtype=np.uint8))

        # Check for key presses
        key = cv2.waitKey(1) & 0xFF
        
        # Exit on pressing the ESC key (ASCII 27)
        if key == 27:
            break
        # Save cheek image on pressing 'S' key
        elif key == ord('s') or key == ord('S'):
            result = save_cheek_image(current_cheek)
            lab_values = save_lab_values(result)
            matching_foundations = find_matching_foundations(lab_values)
            print(matching_foundations)


            print(result)
            # Display save confirmation
            save_notification = np.zeros((100, 400, 3), dtype=np.uint8)
            cv2.putText(save_notification, result, (10, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            cv2.imshow("Save Status", save_notification)
            cv2.waitKey(1500)  # Display for 1.5 seconds
            cv2.destroyWindow("Save Status")

    cap.release()
    cv2.destroyAllWindows()

# --- Tkinter GUI for "Start" button ---
def start_app():
    root.destroy()  # close the start window
    main()          # start the main camera loop

if __name__ == "__main__":
    # Create a simple Tkinter window with a Start button
    root = tk.Tk()
    root.title("Face Detection Start")
    label = tk.Label(root, text="Click 'Start' to begin face detection", font=("Helvetica", 16))
    label.pack(padx=20, pady=20)
    start_button = tk.Button(root, text="Start", command=start_app, font=("Helvetica", 14))
    start_button.pack(padx=20, pady=20)
    root.mainloop()
