import face_recognition
import pickle
import os
import numpy as np

# Load labeled face database
with open("./face_reco_test/labeled_faces.pkl", "rb") as f:
    labeled_data = pickle.load(f)

# Load new image
new_img_path = input("🖼️ Enter the path to the image you want to identify: ").strip()

if not os.path.exists(new_img_path):
    print("❌ File not found!")
    exit()

new_image = face_recognition.load_image_file(new_img_path)
new_face_locations = face_recognition.face_locations(new_image)
new_face_encodings = face_recognition.face_encodings(new_image, new_face_locations)

if not new_face_encodings:
    print("❌ No faces found in the new image.")
    exit()

# Match each detected face
for i, new_encoding in enumerate(new_face_encodings):
    distances = []
    for entry in labeled_data:
        known_encoding = entry["face_encoding"]
        label = entry.get("label", "Unknown")
        dist = np.linalg.norm(known_encoding - new_encoding)
        distances.append((dist, label))

    distances.sort()
    best_match = distances[0]

    print(f"\n🔎 Face #{i + 1}: Closest match: {best_match[1]} (distance: {best_match[0]:.4f})")

    if best_match[0] < 0.6:  # 0.6 is a typical threshold
        print(f"✅ I think this is: {best_match[1]}")
    else:
        print("❌ I don't recognize this face.")
