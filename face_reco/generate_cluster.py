import pickle
import numpy as np
from sklearn.cluster import DBSCAN

# Load face encodings
with open("face_data.pkl", "rb") as f:
    face_data = pickle.load(f)

# Extract encoding vectors
encodings = np.array([record["face_encoding"] for record in face_data])

# Apply DBSCAN clustering
clustering = DBSCAN(eps=0.5, min_samples=2, metric='euclidean')
labels = clustering.fit_predict(encodings)

# Build database with encoding included
db_records = []
for i, label in enumerate(labels):
    entry = {
        "cluster_id": int(label),
        "label": "",  # Optional: assign later via GUI
        "file_name": face_data[i]["file_name"],
        "face_location": face_data[i]["face_location"],
        "face_encoding": face_data[i]["face_encoding"].tolist()  # Convert ndarray to list for pickling or JSON
    }
    db_records.append(entry)

# Save the database
with open("face_clusters_db.pkl", "wb") as f:
    pickle.dump(db_records, f)

# Print summary
print(f"\nSaved {len(db_records)} records to face_clusters_db.pkl")

# Optional: Print cluster summary
cluster_counts = {}
for r in db_records:
    cid = r["cluster_id"]
    cluster_counts[cid] = cluster_counts.get(cid, 0) + 1

print("\nCluster Summary:")
for cid, count in sorted(cluster_counts.items()):
    print(f"Cluster {cid}: {count} face(s)")

