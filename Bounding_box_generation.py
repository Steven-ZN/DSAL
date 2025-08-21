import numpy as np
import os
import cv2
import csv


npy_path = './CAMs'
output_csv = './bounding_boxes.csv'
os.makedirs(os.path.dirname(output_csv), exist_ok=True)

def normalize_cam_to_255(cam):
 
    cam = cam.astype(np.float32)
    min_val, max_val = np.min(cam), np.max(cam)
    if max_val - min_val < 1e-8:
        print("Warning: CAM has zero variance, skip normalization.")
        return cam.astype(np.uint8)
    cam_0_1 = (cam - min_val) / (max_val - min_val)
    cam_0_255 = (cam_0_1 * 255.0).astype(np.uint8)
    return cam_0_255

def get_intensity_threshold(cam_0_255, alpha):

    hist = cv2.calcHist([cam_0_255], [0], None, [256], [0, 256]).ravel()
    cumulative_sum = np.cumsum(hist[::-1])[::-1]
    total_pixels = cam_0_255.size
    alpha_count = (alpha / 100.0) * total_pixels
    T = 255
    for i in range(256):
        if cumulative_sum[i] <= alpha_count:
            T = i
            break
    return T

def get_roi_bbox(cam_0_255, alpha):
   
    T = get_intensity_threshold(cam_0_255, alpha)

    roi_mask = ((cam_0_255 >= T) & (cam_0_255 <= 250)).astype(np.uint8)
    if roi_mask.sum() == 0:
        return None
    contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    h, w = roi_mask.shape[:2]
    x_min_total, y_min_total = w, h
    x_max_total, y_max_total = 0, 0
    for cnt in contours:
        x, y, ww, hh = cv2.boundingRect(cnt)
        x_min_total = min(x_min_total, x)
        y_min_total = min(y_min_total, y)
        x_max_total = max(x_max_total, x + ww)
        y_max_total = max(y_max_total, y + hh)
    if x_max_total <= x_min_total or y_max_total <= y_min_total:
        return None
    return (x_min_total, y_min_total, x_max_total, y_max_total)

def process_cam_file(npy_file, alpha=2.5):
  
    data = np.load(npy_file, allow_pickle=True)
    cam = data.astype(np.float32)
    cam_0_255 = normalize_cam_to_255(cam)
    bbox = get_roi_bbox(cam_0_255, alpha)
    return bbox

def batch_process(folder_path, csv_file, alpha=2.5):

    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Filename", "x_min", "y_min", "x_max", "y_max"])
        for file_name in sorted(os.listdir(folder_path)):
            if file_name.endswith(".npy"):
                npy_file = os.path.join(folder_path, file_name)
                bbox = process_cam_file(npy_file, alpha=alpha)
                if bbox:
                    writer.writerow([file_name] + list(bbox))
                else:
                    writer.writerow([file_name, None, None, None, None])
    print(f"save to：{csv_file}")

if __name__ == "__main__":
    batch_process(npy_path, output_csv, alpha=2.5)
