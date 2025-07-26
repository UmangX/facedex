import pickle
import numpy as np
from sklearn.cluster import DBSCAN
from collections import defaultdict

# Load face database
with open("./face_reco_test/face_database.pkl", "rb") as f:
    face_data = pickle.load(f)

# Get all face encodings
encodings = [entry["face_encoding"] for entry in face_data]

# Use DBSCAN to group similar faces
clustering = DBSCAN(eps=0.6, min_samples=1, metric='euclidean').fit(encodings)

# Assign cluster IDs to each face
for i, entry in enumerate(face_data):
    entry["cluster_id"] = int(clustering.labels_[i])

# Group by cluster
cluster_groups = defaultdict(list)
for entry in face_data:
    cluster_groups[entry["cluster_id"]].append(entry["file_name"])

# Print grouped results
print("\n👥 Face Groups:")
for cluster_id, files in cluster_groups.items():
    print(f"🔹 Group {cluster_id}: {', '.join(files)}")

# Save updated face data with clusters
with open("./face_reco_test/grouped_faces.pkl", "wb") as f:
    pickle.dump(face_data, f)

print(f"\n✅ Grouped face data saved to grouped_faces.pkl")
