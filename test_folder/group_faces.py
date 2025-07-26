import pickle
import face_recognition
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# 📂 Load the database
DB_PATH = "./face_reco_test/face_database.pkl"

with open(DB_PATH, "rb") as db_file:
    face_data = pickle.load(db_file)

print(f"\n🔍 Loaded {len(face_data)} face records from DB.")

# 🧠 Store grouped data
groups = []
used_indexes = set()

# 🧠 Distance threshold (lower = stricter, higher = lenient)
SIMILARITY_THRESHOLD = 0.6  # You can play with this (0.45 - 0.6)

for i, record in enumerate(face_data):
    if i in used_indexes:
        continue

    current_group = [record["file_name"]]
    used_indexes.add(i)

    for j, other in enumerate(face_data):
        if j == i or j in used_indexes:
            continue

        sim = face_recognition.face_distance(
            [record["face_encoding"]],
            other["face_encoding"]
        )[0]

        if sim < SIMILARITY_THRESHOLD:
            current_group.append(other["file_name"])
            used_indexes.add(j)

    groups.append(current_group)

# 🎉 Output result
print("\n📸 Grouped Faces:")
for idx, group in enumerate(groups, 1):
    print(f"Group {idx} ({len(group)} face(s)): {group}")
