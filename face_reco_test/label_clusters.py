import pickle
from collections import defaultdict

# Load grouped faces
with open("./face_reco_test/grouped_faces.pkl", "rb") as f:
    face_data = pickle.load(f)

# Organize by cluster_id
clusters = defaultdict(list)
for entry in face_data:
    clusters[entry["cluster_id"]].append(entry)

# Ask user to label each cluster
print("\n🎨 Time to label your beautiful face clusters!\n")
for cluster_id, entries in clusters.items():
    file_names = [entry["file_name"] for entry in entries]
    print(f"\n🔹 Group {cluster_id} contains: {', '.join(file_names)}")
    label = input(f"👤 Enter name for Group {cluster_id}: ").strip()
    
    for entry in entries:
        entry["label"] = label

# Save labeled data
with open("./face_reco_test/labeled_faces.pkl", "wb") as f:
    pickle.dump(face_data, f)

print("\n✅ All clusters labeled and saved to labeled_faces.pkl 🎉")
