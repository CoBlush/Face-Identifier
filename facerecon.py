import face_recognition
import os
import sys
import cv2
import shutil


def load_target_face(target_image_path):
    """
    Load and encode the target face from the provided image path.
    """
    if not os.path.isfile(target_image_path):
        print(f"Error: Target image '{target_image_path}' does not exist.")
        sys.exit(1)

    image = face_recognition.load_image_file(target_image_path)
    encodings = face_recognition.face_encodings(image)

    if len(encodings) == 0:
        print(f"Error: No faces found in the target image '{target_image_path}'.")
        sys.exit(1)

    return encodings[0]


def scan_images(folder_path, target_encoding, tolerance=0.6):
    """
    Scan the folder for images containing the target face.
    Returns a list of image file paths that contain the target face.
    """
    if not os.path.isdir(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist.")
        sys.exit(1)

    # Supported image extensions
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')
    matched_images = []

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(image_extensions):
                image_path = os.path.join(root, file)
                try:
                    image = face_recognition.load_image_file(image_path)
                    face_locations = face_recognition.face_locations(image)
                    face_encodings = face_recognition.face_encodings(image, face_locations)

                    for encoding in face_encodings:
                        distance = face_recognition.face_distance([target_encoding], encoding)[0]
                        if distance < tolerance:
                            matched_images.append(image_path)
                            print(f"Match found: {image_path} (Distance: {distance:.4f})")
                            break  # Stop checking other faces in the image
                except Exception as e:
                    print(f"Warning: Could not process image '{image_path}'. Error: {e}")

    return matched_images


def prompt_user_action(image_path):
    """
    Prompt the user to decide whether to keep or remove the image.
    Returns True if the image should be removed, False otherwise.
    """
    while True:
        response = input(f"Do you want to remove this image? (y/n): {image_path} ").strip().lower()
        if response in ['y', 'yes']:
            return True
        elif response in ['n', 'no']:
            return False
        else:
            print("Please respond with 'y' or 'n'.")


def main():
    print("=== Face Recognition and Image Management ===")

    # Input target image path
    target_image_path = input("Enter the path to the target face image: ").strip()

    # Load target face encoding
    print("Loading target face...")
    target_encoding = load_target_face(target_image_path)

    # Input folder to scan
    folder_path = input("Enter the path to the folder to scan: ").strip()

    # Scan images
    print("\nScanning images for matches...")
    matched_images = scan_images(folder_path, target_encoding)

    if not matched_images:
        print("No images with the target face were found.")
        sys.exit(0)

    print(f"\nFound {len(matched_images)} image(s) containing the target face.")

    # Optionally, specify a folder to move removed images
    remove_folder = os.path.join(folder_path, "removed_faces")
    os.makedirs(remove_folder, exist_ok=True)

    # Iterate over matched images and prompt user
    for image_path in matched_images:
        remove = prompt_user_action(image_path)
        if remove:
            try:
                # Option 1: Delete the image
                # os.remove(image_path)

                # Option 2: Move the image to 'removed_faces' folder
                shutil.move(image_path, remove_folder)
                print(f"Moved '{image_path}' to '{remove_folder}'.")
            except Exception as e:
                print(f"Error: Could not remove/move image '{image_path}'. Error: {e}")
        else:
            print(f"Kept '{image_path}'.")

    print("\nOperation completed.")


if __name__ == "__main__":
    main()
