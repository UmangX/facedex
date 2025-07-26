import face_recognition
import os
import shutil
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REFERENCE_FOLDER = os.path.join(BASE_DIR, "reference_faces")
UNSORTED_FOLDER = os.path.join(BASE_DIR, "unsorted_faces")
SORTED_FOLDER = os.path.join(BASE_DIR, "sorted_faces")
TOLERANCE = 0.6

print("🧠 Loading reference faces...")

known_encodings = []
known_labels = []

for filename in os.listdir(REFERENCE_FOLDER):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        path = os.path.join(REFERENCE_FOLDER, filename)
        image = face_recognition.load_image_file(path)
        locations = face_recognition.face_locations(image)
        encodings = face_recognition.face_encodings(image, locations)

        if encodings:
            known_encodings.append(encodings[0])
            label = os.path.splitext(filename)[0]
            known_labels.append(label)
            print(f"✅ Loaded: {label}")
        else:
            print(f"⚠️ No face found in reference: {filename}")

print("\n📂 Sorting unsorted faces...")

os.makedirs(SORTED_FOLDER, exist_ok=True)

for filename in os.listdir(UNSORTED_FOLDER):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        path = os.path.join(UNSORTED_FOLDER, filename)
        image = face_recognition.load_image_file(path)
        locations = face_recognition.face_locations(image)
        encodings = face_recognition.face_encodings(image, locations)

        if not encodings:
            print(f"❌ No face in {filename}, skipping.")
            continue

        matched = False
        for face_encoding in encodings:
            results = face_recognition.compare_faces(known_encodings, face_encoding, TOLERANCE)
            if any(results):
                match_index = results.index(True)
                person_name = known_labels[match_index]

                person_folder = os.path.join(SORTED_FOLDER, person_name)
                os.makedirs(person_folder, exist_ok=True)
                shutil.copy(path, os.path.join(person_folder, filename))
                print(f"📦 {filename} → {person_name}")
                matched = True
                break

        if not matched:
            unknown_folder = os.path.join(SORTED_FOLDER, "unknown")
            os.makedirs(unknown_folder, exist_ok=True)
            shutil.copy(path, os.path.join(unknown_folder, filename))
            print(f"❓ {filename} → unknown")

print("\n🎉 Done sorting all faces!")
