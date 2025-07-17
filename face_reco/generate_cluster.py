import pickle
import numpy as np
import hdbscan

# Load face encodings
with open("face_data.pkl", "rb") as f:
    face_data = pickle.load(f)

# Extract encoding vectors
encodings = np.array([record["face_encoding"] for record in face_data])

# Apply HDBSCAN clustering
clusterer = hdbscan.HDBSCAN(min_cluster_size=2, metric='euclidean')
labels = clusterer.fit_predict(encodings)
probabilities = clusterer.probabilities_

# Build database with cluster and encoding info
db_records = []
for i, (label, prob) in enumerate(zip(labels, probabilities)):
    entry = {
        "cluster_id": int(label),
        "label": "",  # Optional: assign later via GUI
        "file_name": face_data[i]["file_name"],
        "face_location": face_data[i]["face_location"],
        "face_encoding": face_data[i]["face_encoding"].tolist(),
        "cluster_confidence": float(prob)  # Optional: for filtering weak assignments
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

