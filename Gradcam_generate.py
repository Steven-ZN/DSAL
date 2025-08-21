import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import amr
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = amr.AMRNet(num_classes=2, pretrained=False).to(device)
model.load_state_dict(torch.load("./AMRNet_weights_AL3.pth"))
model.eval()


base_path = "./BUSI_with_mask/Full_Images"  
train_filename = "train.xlsx"
test_filename = "test.xlsx"
save_path = "./visualized_CAMs/"
os.makedirs(save_path, exist_ok=True)

train_data = pd.read_excel(os.path.join(base_path, train_filename))
test_data = pd.read_excel(os.path.join(base_path, test_filename))


train_data = train_data[train_data['Label'] != "normal"]
test_data = test_data[test_data['Label'] != "normal"]

images_train = [os.path.join(base_path, img) for img in train_data['Image'].tolist()]
labels_train = train_data['Label'].tolist()
images_test = [os.path.join(base_path, img) for img in test_data['Image'].tolist()]
labels_test = test_data['Label'].tolist()

print(f"Train: {len(images_train)}, Test: {len(images_test)}")



def preprocess_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image at {image_path}.")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    original_size = (img.shape[1], img.shape[0])  # (width, height)


    img = img / 255.0
    img = np.transpose(img, (2, 0, 1))  # [C, H, W]
    img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)  # [1, C, H, W]
    return img.to(device), original_size



def apply_colormap_on_image(original_img, cam, alpha=0.5):
    # cam: shape [H, W]
    # original_img: shape [H, W, 3]


    # cam_resized = cv2.resize(cam, (original_img.shape[1], original_img.shape[0]), interpolation=cv2.INTER_LINEAR)
   
    cam_resized = cam


    cam_resized = (cam_resized - cam_resized.min()) / (cam_resized.max() - cam_resized.min() + 1e-8)
    cam_resized = np.uint8(255 * cam_resized)


    heatmap = cv2.applyColorMap(cam_resized, cv2.COLORMAP_JET)

    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)


    superimposed_img = cv2.addWeighted(original_img, 1 - alpha, heatmap, alpha, 0)

    return superimposed_img



for idx, image_path in enumerate(images_train):
    print(f"Processing {idx + 1}/{len(images_train)}: {image_path}")
    basename = os.path.splitext(os.path.basename(image_path))[0]


    original_img = cv2.imread(image_path)
    if original_img is None:
        print(f"Warning: Failed to load {image_path}")
        continue
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

  
    input_image, original_size = preprocess_image(image_path)

    with torch.no_grad():
        if hasattr(model, "generate_spotlight_cam"):
            cam_spotlight = model.generate_spotlight_cam(input_image)
        else:
            raise AttributeError("Model does not have 'generate_spotlight_cam' method.")


    print(f"Raw CAM shape: {cam_spotlight.shape}")  

    cam_spotlight_np = cam_spotlight.squeeze(0).cpu().numpy()

 
    if cam_spotlight_np.ndim == 3:
        cam_spotlight_np = cam_spotlight_np.mean(axis=0)

    print(f"CAM shape after squeeze & mean: {cam_spotlight_np.shape}")  # [13, 20]


    cam_spotlight_np = cv2.resize(
        cam_spotlight_np,
        (original_size[0], original_size[1]),  # (width, height)
        interpolation=cv2.INTER_LINEAR
    )

   
    print(f"CAM shape after resize: {cam_spotlight_np.shape}")  


    if np.isnan(cam_spotlight_np).any():
        print("Warning: CAM contains NaN values!")
        continue

    if cam_spotlight_np.size == 0:
        print("Error: Empty CAM array, skipping save!")
        continue

   
    save_path2 = "./CAMs"
    npy_filename = os.path.join(save_path2, f"{basename}_cam.npy")
    np.save(npy_filename, cam_spotlight_np)


    test_load = np.load(npy_filename)
    print(f"Loaded npy shape: {test_load.shape}, Min: {test_load.min()}, Max: {test_load.max()}")

    colored_cam = apply_colormap_on_image(original_img, cam_spotlight_np)

 
    cam_filename = os.path.join(save_path, f"{basename}_cam.jpg")

    cv2.imwrite(cam_filename, cv2.cvtColor(colored_cam, cv2.COLOR_RGB2BGR))

    print(f"Saved {cam_filename}, {npy_filename}")

print(f"All Spotlight CAMs saved to: {save_path}")
