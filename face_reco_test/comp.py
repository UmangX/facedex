import face_recognition
import os

# Reference image: Bill
ref_image_path = "./face_reco_test/test_folder/bill.jpg"
print("🔍 Checking file path:", os.path.abspath(ref_image_path))
print("📦 File exists?", os.path.isfile(ref_image_path))

if not os.path.isfile(ref_image_path):
    raise FileNotFoundError(f"❌ File not found: {ref_image_path}")

# Load and encode known face (Bill)
ref_image = face_recognition.load_image_file(ref_image_path)
ref_face_locations = face_recognition.face_locations(ref_image)
ref_encodings = face_recognition.face_encodings(ref_image, ref_face_locations)

if not ref_encodings:
    raise Exception("😥 No face found in bill.jpg")

ref_encoding = ref_encodings[0]  # Use the first detected face

# Folder to scan for matches
target_folder = "./face_reco_test/test_folder"

print("\n🧠 Scanning for matching faces...\n")
for file_name in os.listdir(target_folder):
    image_path = os.path.join(target_folder, file_name)

    # Skip comparing with the reference image itself
    if file_name == "bill.jpg":
        continue

    try:
        image = face_recognition.load_image_file(image_path)
        face_locations = face_recognition.face_locations(image)
        encodings = face_recognition.face_encodings(image, face_locations)

        if not encodings:
            print(f"😶 No face found in {file_name}")
            continue

        results = face_recognition.compare_faces(encodings, ref_encoding)

        if any(results):
            print(f"✅ Match found in file: {file_name}")
        else:
            print(f"❌ No match in file: {file_name}")

    except Exception as e:
        print(f"💥 Failed to process {file_name}: {e}")
