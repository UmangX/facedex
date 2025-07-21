import pickle
import face_recognition
import sys
import os
import numpy as np
import hdbscan
import shutil
import json
from PIL import Image

def main():
    if len(sys.argv) != 2:
        print("Usage: python face_cluster.py /path/to/image/folder")
        sys.exit(1)

    folder_path = sys.argv[1]

    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist.")
        sys.exit(1)

    facedx_path = os.path.join(folder_path, "facedx")
    if os.path.exists(facedx_path):
        shutil.rmtree(facedx_path)
    os.makedirs(facedx_path)

    print(f"Created facedx directory at: {facedx_path}")

    print("\n=== Step 1: Generating face encodings ===")
    face_data = generate_encodings(folder_path, facedx_path)

    print("\n=== Step 2: Clustering faces ===")
    db_records = generate_clusters(facedx_path)

    print("\n=== Step 3: Generating thumbnails ===")
    generate_thumbnails(folder_path, facedx_path, db_records)

    print(f"\nComplete! All files saved in: {facedx_path}")

def generate_encodings(folder_path, facedx_path):
    all_face_data = []

    image_files = [f for f in os.listdir(folder_path)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.heic'))]

    if not image_files:
        print("No image files found in the folder.")
        sys.exit(1)

    print(f"Found {len(image_files)} image files to process...")

    for file_name in image_files:
        full_path = os.path.join(folder_path, file_name)
        try:
            image = face_recognition.load_image_file(full_path)
            face_locations = face_recognition.face_locations(image)
            face_encodings = face_recognition.face_encodings(image, face_locations)

            for loc, enc in zip(face_locations, face_encodings):
                face_record = {
                    "file_name": file_name,
                    "face_location": loc,
                    "face_encoding": enc
                }
                all_face_data.append(face_record)

            print(f"  Processed {file_name}: {len(face_encodings)} face(s) found")

        except Exception as e:
            print(f"  Failed to process {file_name}: {e}")

    encodings_path = os.path.join(facedx_path, "face_data.pkl")
    with open(encodings_path, "wb") as f:
        pickle.dump(all_face_data, f)

    print(f"Saved {len(all_face_data)} face encodings to face_data.pkl")
    return all_face_data

def generate_clusters(facedx_path):
    encodings_path = os.path.join(facedx_path, "face_data.pkl")
    with open(encodings_path, "rb") as f:
        face_data = pickle.load(f)

    if len(face_data) == 0:
        print("No face data found for clustering.")
        return []

    encodings = np.array([record["face_encoding"] for record in face_data])
    print(f"Clustering {len(encodings)} face encodings...")

    clusterer = hdbscan.HDBSCAN(min_cluster_size=2, metric='euclidean')
    labels = clusterer.fit_predict(encodings)
    probabilities = clusterer.probabilities_

    db_records = []
    for i, (label, prob) in enumerate(zip(labels, probabilities)):
        entry = {
            "cluster_id": int(label),
            "label": "",
            "file_name": face_data[i]["file_name"],
            "face_location": face_data[i]["face_location"],
            "face_encoding": face_data[i]["face_encoding"].tolist(),
            "cluster_confidence": float(prob)
        }
        db_records.append(entry)

    db_path = os.path.join(facedx_path, "face_clusters_db.pkl")
    with open(db_path, "wb") as f:
        pickle.dump(db_records, f)

    print(f"Saved {len(db_records)} records to face_clusters_db.pkl")

    cluster_counts = {}
    for r in db_records:
        cid = r["cluster_id"]
        cluster_counts[cid] = cluster_counts.get(cid, 0) + 1

    print("\nCluster Summary:")
    for cid, count in sorted(cluster_counts.items()):
        if cid == -1:
            print(f"  Noise (unclustered): {count} face(s)")
        else:
            print(f"  Cluster {cid}: {count} face(s)")

    return db_records

def generate_thumbnails(image_root, facedx_path, db_records):
    thumbnails_path = os.path.join(facedx_path, "thumbnails")
    if os.path.exists(thumbnails_path):
        shutil.rmtree(thumbnails_path)
    os.makedirs(thumbnails_path)

    tracker = []
    json_data = []

    print(f"Generating thumbnails in: {thumbnails_path}")

    for record in db_records:
        cluster_id = record["cluster_id"]
        file_name = record["file_name"]

        if cluster_id == -1 or cluster_id in tracker:
            continue

        try:
            image_path = os.path.join(image_root, file_name)
            if not os.path.exists(image_path):
                print(f"  Image not found: {image_path}")
                continue

            im = Image.open(image_path)

            top, right, bottom, left = record["face_location"]

            face_crop = im.crop((left, top, right, bottom))
            face_thumbnail = face_crop.resize((128, 128))

            thumbnail_filename = f"{cluster_id}.jpg"
            thumbnail_path = os.path.join(thumbnails_path, thumbnail_filename)
            face_thumbnail.save(thumbnail_path)

            current_data = {
                "id": cluster_id,
                "label": "",
                "path": thumbnail_path
            }
            json_data.append(current_data)
            tracker.append(cluster_id)

            print(f"  Generated thumbnail for cluster {cluster_id}")

        except Exception as e:
            print(f"  Failed to generate thumbnail for cluster {cluster_id}: {e}")

    json_path = os.path.join(thumbnails_path, "facedx.json")
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)

    print(f"Generated {len(json_data)} thumbnails and saved metadata to facedx.json")

if __name__ == "__main__":
    main()
