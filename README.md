<p align="center">
  <img src="https://github.com/umangx/facedex/blob/main/facedex.png?raw=true" />
</p>

This is a local version of Google Photos’ face tagging feature.  
It organizes photos by the people in them — no internet or cloud required.

---

### How It Works

- Reads all the images from a folder
- Detects and extracts each face from those images
- Builds a database of face encodings and links them to the images they appear in

---

### Done

- generate encodings for faces in the given folder 
- analyze the encodings for the faces and generate clusters using hdbscan 
- extract faces from the images and generate thumbnails for each cluster 
- combine this functions into single python script for further use in tauri/rust setup 

---

### To Do

- use rust/tauri for making cross-platform desktop program 
- label the thumbnails using frontend and group images based on encoding 
- package this into single executable for use 

---


