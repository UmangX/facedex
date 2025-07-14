import os
import pickle
import face_recognition
from PIL import Image, ImageDraw
import inquirer

# Base path
BASE_DIR = "/Users/synyster7x/projects/local_photos/face_reco_test"
PICKLE_PATH = os.path.join(BASE_DIR, "face_data.pkl")
IMAGE_DIR = os.path.join(BASE_DIR, "test_folder")

# Load face data
with open(PICKLE_PATH, "rb") as file:
    data = pickle.load(file)

# Get unique image file names
image_names = list(set(entry["file_name"] for entry in data))

# Prompt user to select an image
questions = [
    inquirer.List(
        "selected_image",
        message="Select an image to view faces",
        choices=image_names
    )
]
answers = inquirer.prompt(questions)
selected_image_name = answers["selected_image"]

# Construct full image path
image_path = os.path.join(IMAGE_DIR, selected_image_name)

# Load the image
image = face_recognition.load_image_file(image_path)

# Get face locations for selected image
face_locations = [entry["face_location"] for entry in data if entry["file_name"] == selected_image_name]

# Draw rectangles on the image
pil_image = Image.fromarray(image)
draw = ImageDraw.Draw(pil_image)
for (top, right, bottom, left) in face_locations:
    draw.rectangle([left, top, right, bottom], outline="red", width=3)

# Show image
pil_image.show()

