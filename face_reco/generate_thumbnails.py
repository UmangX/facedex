import os
import sys
import pickle
from PIL import Image

if len(sys.argv) != 2:
    print("Usage: python generate_one_thumbnail_per_cluster.py /path/to/image/folder")
    sys.exit(1)

# Folder containing the actual image files
image_root = sys.argv[1]

# Load clustered face data
with open("face_clusters_db.pkl", "rb") as f:
    db_records = pickle.load(f)

# Output folder for thumbnails
output_dir = "thumbnails"
os.makedirs(output_dir, exist_ok=True)

# Track one representative per cluster
cluster_thumbnails = {}

for record in db_records:
    cluster_id = record["cluster_id"]
    if cluster_id not in cluster_thumbnails:
        cluster_thumbnails[cluster_id] = record

# Generate one thumbnail per cluster
for cluster_id, record in cluster_thumbnails.items():
    # Construct full path to image
    file_path = os.path.join(image_root, record["file_name"])
    top, right, bottom, left = record["face_location"]

    try:
        img = Image.open(file_path)
        face_img = img.crop((left, top, right, bottom))
        face_img = face_img.resize((128, 128))

        # Determine output file name
        name = f"cluster_{cluster_id}.jpg" if cluster_id != -1 else "noise.jpg"
        out_path = os.path.join(output_dir, name)

        face_img.save(out_path)
        print(f"Saved thumbnail for cluster {cluster_id}: {out_path}")

    except Exception as e:
        print(f"Error processing {file_path}: {e}")

