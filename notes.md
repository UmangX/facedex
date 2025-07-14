# Local Face Tagging 

This is a local version of Google Photos’ face tagging feature.  
It organizes photos by the people in them — no internet or cloud required.

---

### How It Works

- Reads all the images from a folder
- Detects and extracts each face from those images
- Builds a database of face encodings and links them to the images they appear in

---

### Done

- For single images:
  - Locate faces
  - Get face encodings
  - Compare encodings to check similarity

---

### To Do

- Create a full database for face encodings
- Add manual or automatic labels (e.g. "John", "Mom", etc.)
- Remove duplicate/similar encodings per label
- Improve face matching accuracy

---


