#!/usr/bin/python3
import sys
from joblib.logger import shutil
import numpy as np
import face_recognition
import os
import hdbscan
import pickle
import json
from PIL import Image

def generate_encoding(path):
    face_data = []
    for file_name in os.listdir(path):
        full_path = os.path.join(path, file_name)
        if not file_name.lower().endswith(('.png', '.jpg', '.jpeg','.HEIC')):
            continue  # skip non-image files
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
                face_data.append(face_record)
            print(f"Processed {file_name}: {len(face_encodings)} face(s) found.")
        except Exception as e:
            print(f"Failed to process {file_name}: {e}")
    return face_data

def generate_clusters(face_data,path):
    encodings = np.array([record["face_encoding"] for record in face_data])
    clusterer = hdbscan.HDBSCAN(min_cluster_size=2, metric='euclidean')
    labels = clusterer.fit_predict(encodings)
    probabilities = clusterer.probabilities_
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
    with open(os.path.join(path,"face_clusters_db.pkl"), "wb") as f:
        pickle.dump(db_records, f)
    print(f"\nSaved {len(db_records)} records to face_clusters_db.pkl")
    cluster_counts = {}
    for r in db_records:
        cid = r["cluster_id"]
        cluster_counts[cid] = cluster_counts.get(cid, 0) + 1
    print("\nCluster Summary:")
    for cid, count in sorted(cluster_counts.items()):
        print(f"Cluster {cid}: {count} face(s)")

def generate_thumbs(folder_path):
    facedex_path = os.path.join(folder_path,"facedex")
    with open(os.path.join(facedex_path,"face_clusters_db.pkl"), "rb") as f:
        db_records = pickle.load(f)
    tracker = []
    def handle_thumbnaildir():
        if os.path.exists(os.path.join(facedex_path,"thumbnails")):
            shutil.rmtree(os.path.join(facedex_path,"thumbnails"))

    handle_thumbnaildir()
    os.mkdir(os.path.join(facedex_path,"thumbnails"))

    json_data = []

    for i in db_records:
        id = i["cluster_id"]
        file_name = i["file_name"]
        if id not in tracker:
            # Generate the thumbnails
            im = Image.open(os.path.join(folder_path,file_name))
            top,right,bottom,left = i["face_location"]
            id_thumbnail = im.crop((left,top,right,bottom))
            id_thumbnail.resize((128,128))
            # Handle thumbnails folder and JSON file
            id_thumbnail.save(facedex_path+"/thumbnails/"+str(id)+".jpg")
            current_data = {"id" : id , "label":" ","path" : facedex_path+"/thumbnails/"+str(id)+".jpg"}
            json_data.append(current_data)
            tracker.append(id)
            print(f"working on id : {id}")


    with open(facedex_path+"/thumbnails/"+"facedex.json","w") as file:
        json.dump(json_data,file,indent=2)


def main():
    folder_path = ""
    if len(sys.argv) < 2:
        print("provide the folder")
    else:
        folder_path = sys.argv[1];
    face_encoding = generate_encoding(folder_path)
    try:
        os.mkdir(folder_path+"/facedex")
    except FileExistsError:
        shutil.rmtree(folder_path+"/facedex",ignore_errors=True)
        os.mkdir(folder_path+"/facedex")
    facedex_path = os.path.join(folder_path,"facedex")
    generate_clusters(face_encoding,facedex_path)
    generate_thumbs(folder_path)
if __name__ == "__main__":
    main()
