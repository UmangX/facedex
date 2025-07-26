from PIL import Image
import os

# Updated paths
GROUPED_FOLDER = "./test_folder/sorted_faces"
THUMBNAIL_FOLDER = "./thumbnails"
THUMBNAIL_SIZE = (150, 150)

os.makedirs(THUMBNAIL_FOLDER, exist_ok=True)

print("🖼️ Generating thumbnails for face groups...")

for group_name in os.listdir(GROUPED_FOLDER):
    group_path = os.path.join(GROUPED_FOLDER, group_name)

    if not os.path.isdir(group_path):
        continue

    for file in os.listdir(group_path):
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(group_path, file)
            try:
                image = Image.open(image_path)
                image.thumbnail(THUMBNAIL_SIZE)
                save_path = os.path.join(THUMBNAIL_FOLDER, f"{group_name}.jpg")
                image.save(save_path)
                print(f"✅ Thumbnail created for {group_name}")
                break
            except Exception as e:
                print(f"❌ Error in {group_name}: {e}")
                break

print("\n🌈 All thumbnails saved in:", THUMBNAIL_FOLDER)
