import os

print("📁 Folders I see here:")
print(os.listdir("."))

print("\n🧪 Does 'reference_faces' exist?", os.path.exists("../reference_faces"))
print("🧪 Does 'unsorted_faces' exist?", os.path.exists("../unsorted_faces"))
