import os
import sys
import pickle
from PIL import Image
import json 
import shutil

if len(sys.argv) != 2:
    print("Usage: python generate_one_thumbnail_per_cluster.py /path/to/image/folder")
    sys.exit(1)
# Folder containing the actual image files
image_root = sys.argv[1]
# Load clustered face data
with open("face_clusters_db.pkl", "rb") as f:
    db_records = pickle.load(f)
pwd = os.getcwd()
thumb_path = os.path.join(pwd,"thumbnails")
print(f"current working directory : {pwd}")
tracker = []

def handle_thumbnaildir():
    if os.path.exists(os.path.join(pwd,"thumbnails")):
        shutil.rmtree(os.path.join(pwd,"thumbnails"))

handle_thumbnaildir()
os.mkdir(os.path.join(pwd,"thumbnails")) 

json_data = []

for i in db_records:
    id = i["cluster_id"]
    file_name = i["file_name"]
    if id not in tracker:
        # Generate the thumbnails 
        im = Image.open(os.path.join(image_root,file_name))
        top,right,bottom,left = i["face_location"]
        id_thumbnail = im.crop((left,top,right,bottom))
        id_thumbnail.resize((128,128))
        # Handle thumbnails folder and JSON file
        id_thumbnail.save(pwd+"/thumbnails/"+str(id)+".jpg")
        current_data = {"id" : id , "label":" ","path" : pwd+"/thumbnails/"+str(id)+".jpg"}
        json_data.append(current_data)
        tracker.append(id)
        print(f"working on id : {id}") 


with open(pwd+"/thumbnails/"+"facedex.json","w") as file:
    json.dump(json_data,file,indent=2)
