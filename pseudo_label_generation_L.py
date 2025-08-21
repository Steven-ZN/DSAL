import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3,4,5,6,7'

import numpy as np
import torch
from PIL import Image
from transformers import SamModel, SamProcessor
import cv2
import csv
from utility import VOCPalette

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SamModel.from_pretrained("facebook/sam-vit-huge").to(device)
processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")


# Filename, x_min, y_min, x_max, y_max
roi_csv = "./bounding_boxes.csv"

image_dir = "./selected_images"

pseudo_labels_path = "./pseudolabels"
os.makedirs(pseudo_labels_path, exist_ok=True)


palette = VOCPalette(nb_class=2)


def segment_image_with_sam(image_np, box):
    
    
    x_min, y_min, x_max, y_max = box
    w = x_max - x_min
    h = y_max - y_min

    
    inputs = processor(
        images=image_np,
        input_boxes=[[[x_min, y_min, x_min + w, y_min + h]]],  # 二层list
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs, multimask_output=False)

    
    masks = processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu()
    )
    
    mask = masks[0][0].cpu().numpy().astype(np.uint8)
    return mask


with open(roi_csv, "r", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        img_name = os.path.splitext(row["Filename"])[0] + ".jpg"
        if (not row["x_min"]) or (not row["x_max"]) or (not row["y_min"]) or (not row["y_max"]):
            print(f"Warning: invalid bbox for {row['Filename']} => skip.")
            continue  

        try:
            x_min = int(float(row["x_min"]))
            y_min = int(float(row["y_min"]))
            x_max = int(float(row["x_max"]))
            y_max = int(float(row["y_max"]))
        except ValueError:
            print(f"Warning: cannot parse bbox for {row['Filename']} => skip.")
            continue
        
        image_path = os.path.join(image_dir, img_name)
        if not os.path.exists(image_path):
            print(f"Warning: {image_path} not found. Skip.")
            continue
        img_pil = Image.open(image_path).convert("RGB")
        image_np = np.array(img_pil)

        
        mask = segment_image_with_sam(image_np, [x_min, y_min, x_max, y_max])

        
        
        
        mask = mask.squeeze(0)  

        
        mask_pal = palette.genlabelpal(mask)
        save_name = os.path.splitext(img_name)[0] + ".png"
        mask_pal.save(os.path.join(pseudo_labels_path, save_name))

        print(f"[Done] {img_name} => saved pseudo label: {save_name}")
