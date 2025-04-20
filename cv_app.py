import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import ttk, messagebox  # Added messagebox import
import os
import json
from datetime import datetime
from utils.extract import extract_lab_from_image, find_matching_foundations
import io
import base64

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

def pil_to_tk(image):
    with io.BytesIO() as output:
        image.save(output, format="PNG")
        data = output.getvalue()
    b64_data = base64.b64encode(data)
    return tk.PhotoImage(data=b64_data)

# Left cheek landmark indices
left_cheek_indices = [120, 117, 187, 203]

def warp_cheek_to_square(cheek_points, image):
    if len(cheek_points) != 4:
        return None  # Ensure exactly 4 points

    target_square = np.array([[0, 0], [99, 0], [99, 99], [0, 99]], dtype=np.float32)
    cheek_points = np.array(cheek_points, dtype=np.float32)
    M = cv2.getPerspectiveTransform(cheek_points, target_square)
    warped_cheek = cv2.warpPerspective(image, M, (100, 100))
    return warped_cheek

def process_frame(image):
    h, w, _ = image.shape
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)
    annotated_image = image.copy()
    cheek_region = None

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=1)
            )
            points = [[int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)] 
                      for i in left_cheek_indices]

            if len(points) == 4:
                points = np.array(points)
                hull = cv2.convexHull(points)
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillConvexPoly(mask, hull, 255)
                cheek_full = cv2.bitwise_and(image, image, mask=mask)
                x, y, w_box, h_box = cv2.boundingRect(hull)
                cheek_region = cheek_full[y:y+h_box, x:x+w_box]

                if cheek_region is not None and cheek_region.size > 0:
                    warped_cheek = warp_cheek_to_square(points, image)
                    if warped_cheek is not None:
                        cheek_region = warped_cheek
    return annotated_image, cheek_region

def save_cheek_image(cheek_image):
    save_dir = "saved_cheeks"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{save_dir}/cheek_{timestamp}.jpg"
    if cheek_image is not None and cheek_image.size > 0:
        cv2.imwrite(filename, cheek_image)
        return filename
    else:
        return "No valid cheek image to save"

def save_lab_values(cheek_image):
    lab_values = extract_lab_from_image(cheek_image)
    lab_dir = "lab_values"
    if not os.path.exists(lab_dir):
        os.makedirs(lab_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    lab_filename = f"{lab_dir}/lab_{timestamp}.json"
    if lab_values:
        with open(lab_filename, 'w') as f:
            json.dump({"lab_values": lab_values}, f, indent=4)
        print(f"LAB values saved to {lab_filename}")
    if lab_values:
        return lab_filename, lab_values
    else:
        return "No valid lab values to save"

def show_foundation_matches_window(matching_foundations):
    window = tk.Toplevel()
    window.attributes("-fullscreen", True)
    window.configure(bg="white")

    def exit_app():
        window.destroy()

    # Title
    title = tk.Label(window, text="YOUR SHADE MATCH", font=("Georgia", 32, "bold"), bg="white")
    title.pack(pady=(40, 20))

    # Section frame
    section = tk.Frame(window, bg="white")
    section.pack()

    def create_foundation_column(frame, title_text, match_key, shade_label):
        col = tk.Frame(frame, bg="white")
        tk.Label(col, text=title_text, font=("Helvetica", 14, "bold"), bg="white").pack()

        foundation = matching_foundations.get(match_key)
        if foundation is None:
            tk.Label(col, text="Not available", font=("Georgia", 16, "bold"),
                     fg="#cb4d4d", bg="white").pack(pady=10)
            return col

        img_path = foundation[3]
        try:
            img = Image.open(img_path).resize((100, 130))
            photo = pil_to_tk(img)
            img_label = tk.Label(col, image=photo, bg="white")
            img_label.image = photo  # Prevent garbage collection
            img_label.pack()
        except Exception as e:
            tk.Label(col, text="Image error", font=("Georgia", 12),
                     fg="#cb4d4d", bg="white").pack(pady=10)

        tk.Label(col, text=foundation[0], font=("Georgia", 16, "bold"),
                 fg="#cb4d4d", bg="white").pack()
        tk.Label(col, text=shade_label, font=("Georgia", 12, "italic"),
                 fg="#cb4d4d", bg="white").pack()
        return col

    # Top row (best matches)
    top_row = tk.Frame(section, bg="white")
    top_row.pack(pady=20)

    create_foundation_column(top_row, "for dry to normal skin", "dry_best", "neutral tone").pack(side="left", padx=50)
    create_foundation_column(top_row, "for oily to normal skin", "oily_best", "warm tone").pack(side="left", padx=50)

    # Note
    tk.Label(section,
             text="If you are acidic or tend to oxidize, try a lighter shade than your shade match:",
             font=("Helvetica", 11), bg="white", wraplength=700).pack(pady=(30, 10))

    # Bottom row (lighter matches)
    bottom_row = tk.Frame(section, bg="white")
    bottom_row.pack(pady=10)

    create_foundation_column(bottom_row, "", "dry_lighter", "neutral light").pack(side="left", padx=50)
    create_foundation_column(bottom_row, "", "oily_lighter", "warm light").pack(side="left", padx=50)

    # Action buttons
    button_frame = tk.Frame(window, bg="white")
    button_frame.pack(pady=(40, 20))

    exit_button = tk.Button(button_frame, text="EXIT", font=("Helvetica", 12), bg="white", fg="gray",
                            relief="flat", command=exit_app)
    exit_button.pack()

    window.mainloop()

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Camera could not be opened.")
        return

    # Create a Tkinter window to display the video feed
    video_window = tk.Tk()
    video_window.attributes("-fullscreen", True)
    video_window.title("Live Face Mesh")

    # Label to show video frames
    video_label = tk.Label(video_window)
    video_label.pack(expand=True, fill="both")

    # Use a mutable container for the current cheek image
    current_cheek = [None]

    def save_button_callback():
        if current_cheek[0] is None:
            print("No cheek image available.")
            return
        result = save_cheek_image(current_cheek[0])
        lab_filename, _ = save_lab_values(result)
        print(lab_filename)
        matching_foundations = find_matching_foundations(lab_filename)
        print(matching_foundations)
        show_foundation_matches_window(matching_foundations)

    # Button to trigger saving the cheek image
    save_button = tk.Button(
        video_window,
        text="Save Cheek Image",
        font=("Helvetica", 14),
        bg="#bb7b3f",
        fg="white",
        command=save_button_callback
    )
    save_button.pack(side="bottom", pady=20)

    def update_loop():
        ret, frame = cap.read()
        if ret:
            annotated, cheek = process_frame(frame)
            current_cheek[0] = cheek

            # Convert the annotated frame for display in Tkinter
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            im = Image.fromarray(annotated_rgb)
            imgtk = ImageTk.PhotoImage(image=im)
            video_label.imgtk = imgtk  # Prevent garbage collection
            video_label.config(image=imgtk)
        else:
            print("Error: Frame not read properly.")
            cap.release()
            video_window.destroy()
            return

        # Schedule the next frame update after 15ms
        video_window.after(15, update_loop)

    update_loop()
    video_window.mainloop()

def start_app():
    root.destroy()  # close the start window
    main()          # start the main camera loop

if __name__ == "__main__":
    root = tk.Tk()
    root.attributes("-fullscreen", True)
    root.title("Liquid Foundation Shade Matching")
    
    # Load the original background image
    bg_image = Image.open("bg.jpg")
    
    # Create a canvas that fills the window
    canvas = tk.Canvas(root, highlightthickness=0)
    canvas.pack(fill="both", expand=True)
    
    # Place the initial background image on the canvas
    bg_photo = ImageTk.PhotoImage(bg_image)
    bg_img_id = canvas.create_image(0, 0, image=bg_photo, anchor="nw")
    canvas.bg_photo = bg_photo  # Keep a reference to prevent garbage collection

    # Add the white text (initial position; will update on resize)
    text_id = canvas.create_text(
        root.winfo_width() // 2, 300,
        text="LIQUID\nFOUNDATION\nSHADE\nMATCHING",
        fill="brown",
        font=("Georgia", 38, "bold"),
        justify="center"
    )

    # Define the function that resizes the background and adjusts text/button positions
    def resize_bg(event):
        new_width = event.width
        new_height = event.height

        # Resize and update the background image
        resized_image = bg_image.resize((new_width, new_height), Image.LANCZOS)
        new_bg_photo = ImageTk.PhotoImage(resized_image)
        canvas.bg_photo = new_bg_photo  # update reference
        canvas.itemconfig(bg_img_id, image=new_bg_photo)

        # Center the text; adjust vertical position based on canvas height
        canvas.coords(text_id, new_width / 2, new_height * 0.3)
        
        # Also move the button if it's already created
        if hasattr(canvas, "button_window"):
            canvas.coords(canvas.button_window, new_width / 2, new_height * 0.8)

    canvas.bind("<Configure>", resize_bg)

    # Add custom styled button
    def on_click():
        start_app()

    button = tk.Button(
        root,
        text="Match your Shade  ➔",
        font=("Helvetica", 24, "bold"),  # increased font size
        bg="#bb7b3f",
        fg="white",
        bd=0,
        activebackground="#a56a32",
        activeforeground="white",
        relief="flat",
        padx=20,  # extra horizontal padding
        pady=10,  # extra vertical padding
        command=on_click
    )

    # Place the button on the canvas using a window and tag it for repositioning
    canvas.button_window = canvas.create_window(
        root.winfo_width() // 2, 680,
        window=button
    )
    
    root.mainloop()
