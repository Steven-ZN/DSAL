import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3,4,5,6,7'

import numpy as np
import torch
from PIL import Image
from transformers import SamModel, SamProcessor
import cv2
from utility import VOCPalette

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_dir = "./unselected_images"  
cam_dir   = "./CAMs"  
pseudo_dir = "./unselected_images_pseudo"
os.makedirs(pseudo_dir, exist_ok=True)


model = SamModel.from_pretrained("facebook/sam-vit-huge").to(device)
processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")

palette = VOCPalette(nb_class=2)



def normalize_cam_to_255(cam: np.ndarray) -> np.ndarray:
    cam = cam.astype(np.float32)
    min_val, max_val = np.min(cam), np.max(cam)
    if max_val - min_val < 1e-8:
        return cam.astype(np.uint8)
    cam = (cam - min_val) / (max_val - min_val)
    return (cam * 255.).astype(np.uint8)


def get_intensity_threshold(cam_0_255: np.ndarray, alpha: float = 2.5) -> int:
    hist = cv2.calcHist([cam_0_255], [0], None, [256], [0, 256]).ravel()
    cum = np.cumsum(hist[::-1])[::-1]  
    alpha_count = cam_0_255.size * alpha / 100.0
    for i in range(256):
        if cum[i] <= alpha_count:
            return i
    return 255


def get_roi_bbox_from_cam(cam_arr: np.ndarray, alpha: float = 2.5):
    cam_255 = normalize_cam_to_255(cam_arr)
    T = get_intensity_threshold(cam_255, alpha)
    mask = ((cam_255 >= T) & (cam_255 <= 250)).astype(np.uint8)
    if mask.sum() == 0:
        return None
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    h, w = mask.shape[:2]
    x_min, y_min, x_max, y_max = w, h, 0, 0
    for cnt in contours:
        x, y, ww, hh = cv2.boundingRect(cnt)
        x_min, y_min = min(x_min, x), min(y_min, y)
        x_max, y_max = max(x_max, x + ww), max(y_max, y + hh)
    if x_max <= x_min or y_max <= y_min:
        return None
    return [x_min, y_min, x_max, y_max]


def load_cam_and_get_bbox(base_name: str, alpha: float = 2.5):
    
    for suffix in (".npy", "_cam.npy"):
        cam_path = os.path.join(cam_dir, base_name + suffix)
        if os.path.exists(cam_path):
            cam_arr = np.load(cam_path, allow_pickle=True)
            if cam_arr.ndim == 3:
                
                cam_arr = cam_arr.max(axis=0)
            bbox = get_roi_bbox_from_cam(cam_arr, alpha)
            return bbox
    return None



def sam_segment(image_np: np.ndarray, bbox):
    x_min, y_min, x_max, y_max = bbox
    inputs = processor(
        images=image_np,
        input_boxes=[[[x_min, y_min, x_max, y_max]]],
        return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        outputs = model(**inputs, multimask_output=False)
    masks = processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu()
    )
    return masks[0][0].cpu().numpy().astype(np.uint8)


for img_file in sorted(os.listdir(image_dir)):
    if not img_file.lower().endswith((".jpg", ".png", ".jpeg")):
        continue
    base_name = os.path.splitext(img_file)[0]

    
    bbox = load_cam_and_get_bbox(base_name)
    if bbox is None:
        print(f"[Skip] No bbox for {img_file}")
        continue

    
    img_path = os.path.join(image_dir, img_file)
    if not os.path.exists(img_path):
        print(f"[Skip] Image {img_file} not found")
        continue
    image_np = np.array(Image.open(img_path).convert("RGB"))

    
    mask = sam_segment(image_np, bbox)
    mask_pal = palette.genlabelpal(mask.squeeze(0))

    
    save_name = base_name + ".png"
    mask_pal.save(os.path.join(pseudo_dir, save_name))
    print(f"[Done] {img_file} -> {save_name}")
