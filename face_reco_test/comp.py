import face_recognition
from face_recognition.api import face_encodings
import os

# Load and encode known face (ash)
ash_image = face_recognition.load_image_file("./test_folder/ash.jpg")
known_face_loc = face_recognition.face_locations(ash_image)
ash_encoding = face_recognition.face_encodings(ash_image, known_face_loc)

if not ash_encoding:
    raise Exception("No face found in ash.jpg")

ash_encoding = ash_encoding[0]  # Get the first (and likely only) face encoding

# Scan other images in the folder
file_names = os.listdir("./test_folder")
for file_name in file_names:
    comp_name = './test_folder/' + file_name

    # Skip comparing with ash.jpg itself
    if file_name == "ash.jpg":
        continue

    sec_image = face_recognition.load_image_file(comp_name)
    sec_loc = face_recognition.face_locations(sec_image)
    sec_enc = face_encodings(sec_image, sec_loc)

    if not sec_enc:
        continue  # No face found in image, skip

    result = face_recognition.compare_faces(sec_enc, ash_encoding)
    if any(result):
        print(f"✅ Found face in file: {file_name}")
