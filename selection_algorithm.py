import cv2
import numpy as np
import os
import shutil


folder_path = "./visualized_CAMs"


selected_folder = "./selected_images"
if not os.path.exists(selected_folder):
    os.makedirs(selected_folder)

max_contours_allowed = 2

min_area = 500
max_area = 50000
min_circularity = 0.4

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))

lower_red1 = np.array([0, 50, 50])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 50, 50])
upper_red2 = np.array([180, 255, 255])

lower_yellow = np.array([15, 80, 80])   
upper_yellow = np.array([35, 255, 255]) 

def process_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"unable to load image: {image_path}")
        return False

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)

    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    mask = cv2.bitwise_or(mask_red, mask_yellow)


    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)


    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > max_contours_allowed:

        return False


    valid_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area < min_area or area > max_area:
            continue

        # circularity = 4π * (area / perimeter^2)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4.0 * np.pi * (area / (perimeter * perimeter))


        if circularity < min_circularity:
            continue

        valid_contours.append(cnt)

    if not valid_contours:
        return False

    all_points = []
    for vc in valid_contours:
        
        all_points.extend(vc.reshape(-1, 2))
    all_points = np.array(all_points)
    x_min, y_min = np.min(all_points, axis=0)
    x_max, y_max = np.max(all_points, axis=0)
    box_w = x_max - x_min
    box_h = y_max - y_min
    coverage_area = box_w * box_h


    img_area = img.shape[0] * img.shape[1]
    coverage_ratio = coverage_area / float(img_area)
    if coverage_ratio > 0.5:

        return False
    return True


if __name__ == "__main__":
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp', '.tif')):
            file_path = os.path.join(folder_path, filename)
            keep = process_image(file_path)
            if keep:
                print(f"[save] {filename}")
                shutil.copy(file_path, selected_folder)
            else:
                print(f"[exclude] {filename}")
