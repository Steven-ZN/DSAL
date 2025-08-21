import os
import shutil


src_folder = "./selected_images_ROI"


ref_folder = "./BUSI/train"


dest_folder = "./unselected_images"
if not os.path.exists(dest_folder):
    os.makedirs(dest_folder)


src_base_set = set()

for filename in os.listdir(src_folder):
    if filename.endswith("_cam.jpg"):
        base_name = filename[:-8]
        src_base_set.add(base_name)


for filename in os.listdir(ref_folder):
    if filename.endswith(".png"):
        base_name = filename[:-4]
        if base_name not in src_base_set:
            src_path = os.path.join(ref_folder, filename)
            dest_path = os.path.join(dest_folder, filename)
            shutil.copy(src_path, dest_path)
            print(f"[copy] {filename} => {dest_folder} (base_name={base_name})")


def delete_files_with_normal(target_folder):
    for filename in os.listdir(target_folder):
        if "normal" in filename:
            file_path = os.path.join(target_folder, filename)
            try:
                os.remove(file_path)
                print(f"[delete] {file_path}")
            except Exception as e:
                print(f"[error] cannot delete {file_path}: {e}")

delete_files_with_normal("./selected_images_ROI")
delete_files_with_normal("./unselected_images")
