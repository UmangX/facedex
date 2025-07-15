import pickle
import numpy as np
from sklearn.cluster import DBSCAN
import os

# Load face encodings
with open("face_data.pkl", "rb") as f:
    face_data = pickle.load(f)

# Extract encoding vectors
encodings = np.array([record["face_encoding"] for record in face_data])

# Apply DBSCAN
# eps: distance threshold (tweak as needed)
# min_samples: how many faces needed to form a cluster
clustering = DBSCAN(eps=0.5, min_samples=2, metric='euclidean')
labels = clustering.fit_predict(encodings)

# Add labels to records
for i, label in enumerate(labels):
    face_data[i]["cluster_id"] = int(label)  # -1 means noise

# Save clustered data
with open("clustered_faces.pkl", "wb") as f:
    pickle.dump(face_data, f)

# Print summary
label_counts = {}
for label in labels:
    label_counts[label] = label_counts.get(label, 0) + 1

print("\nCluster Summary:")
for label, count in sorted(label_counts.items()):
    label_name = f"Cluster {label}" if label != -1 else "Noise (-1)"
    print(f"{label_name}: {count} faces")

# Optional: Save per-cluster file listing
os.makedirs("clusters", exist_ok=True)
clusters = {}
for item in face_data:
    cluster_id = item["cluster_id"]
    clusters.setdefault(cluster_id, []).append(item["file_name"])

for cluster_id, files in clusters.items():
    fname = f"clusters/cluster_{cluster_id}.txt"
    with open(fname, "w") as f:
        for fn in files:
            f.write(fn + "\n")

