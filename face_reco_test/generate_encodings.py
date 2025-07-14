import pickle
import face_recognition
import numpy as np
import sys
import os

if len(sys.argv) != 2:
    print("Error: Provide a folder path for generating encodings.")
    sys.exit(1)

folder_path = sys.argv[1]

if __name__ == "__main__":
    all_face_data = []

    for file_name in os.listdir(folder_path):
        full_path = os.path.join(folder_path, file_name)
        if not file_name.lower().endswith(('.png', '.jpg', '.jpeg','.HEIC')):
            continue  # skip non-image files
        try:
            image = face_recognition.load_image_file(full_path)
            face_locations = face_recognition.face_locations(image)
            face_encodings = face_recognition.face_encodings(image, face_locations)

            for loc, enc in zip(face_locations, face_encodings):
                face_record = {
                    "file_name": file_name,
                    "face_location": loc,
                    "face_encoding": enc
                }
                all_face_data.append(face_record)

            print(f"Processed {file_name}: {len(face_encodings)} face(s) found.")
        except Exception as e:
            print(f"Failed to process {file_name}: {e}")

    # Save all encodings
    with open("face_data.pkl", "wb") as f:
        pickle.dump(all_face_data, f)

    print(f"\nSaved {len(all_face_data)} face encodings to face_data.pkl")
