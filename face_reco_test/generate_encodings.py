import os
import face_recognition
import pickle

# 💼 Folder that holds all your images
FOLDER_PATH = "./face_reco_test/test_folder"
DB_FILE = "./face_reco_test/face_database.pkl"

# 🔐 This will hold all face data records
face_database = []

print("🧠 Starting FaceDex Database Generation...\n")

for file_name in os.listdir(FOLDER_PATH):
    image_path = os.path.join(FOLDER_PATH, file_name)

    # ⚠️ Skip non-image files
    if not file_name.lower().endswith(('.jpg', '.jpeg', '.png', '.heic')):
        continue

    try:
        # 📸 Load the image
        image = face_recognition.load_image_file(image_path)

        # 🔍 Find face locations and encodings
        face_locations = face_recognition.face_locations(image, model="cnn")  # for better accuracy
        face_encodings = face_recognition.face_encodings(image, face_locations)

        if not face_encodings:
            print(f"😶 No face found in {file_name}")
            continue

        # 🏷️ Label: filename without extension (can improve later)
        label = os.path.splitext(file_name)[0]

        for encoding, loc in zip(face_encodings, face_locations):
            face_record = {
                "file_name": file_name,
                "label": label,
                "face_location": loc,
                "face_encoding": encoding
            }
            face_database.append(face_record)

        print(f"✅ Processed {file_name}: {len(face_encodings)} face(s) found.")

    except Exception as e:
        print(f"💥 Failed to process {file_name}: {e}")

# 💾 Save to face_database.pkl
with open(DB_FILE, "wb") as db_file:
    pickle.dump(face_database, db_file)

print(f"\n💽 Saved {len(face_database)} face records to {DB_FILE}")
