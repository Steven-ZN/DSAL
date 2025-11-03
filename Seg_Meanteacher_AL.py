import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'

import torch
import numpy as np
import random
from dataloader1 import BreastCancerSegmentationPseudo, BreastCancerSegmentation
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import pandas as pd
import copy
import itertools
import shutil  

import torchvision.models as models
import torch.nn.functional as F


is_cuda_available = torch.cuda.is_available()
device = torch.device("cuda" if is_cuda_available else "cpu")

seed_value = 42
torch.manual_seed(seed_value)
torch.cuda.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)
np.random.seed(seed_value)
random.seed(seed_value)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def calculate_f1(pred, target):
    
    pred = torch.argmax(pred, dim=1)
    pred_tumor = (pred == 1).float()
    target_tumor = (target == 1).float()
    true_positive = (pred_tumor * target_tumor).sum(dim=(1, 2))
    false_positive = (pred_tumor * (1 - target_tumor)).sum(dim=(1, 2))
    false_negative = ((1 - pred_tumor) * target_tumor).sum(dim=(1, 2))
    precision = true_positive / (true_positive + false_positive + 1e-6)
    recall = true_positive / (true_positive + false_negative + 1e-6)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
    f1[true_positive == 0] = 0
    return f1.mean()

def calculate_iou(pred, target):
    
    pred = torch.argmax(pred, dim=1)
    pred_tumor = (pred == 1).float()
    target_tumor = (target == 1).float()
    intersection = (pred_tumor * target_tumor).sum(dim=(1, 2))
    union = pred_tumor.sum(dim=(1, 2)) + target_tumor.sum(dim=(1, 2)) - intersection
    iou = intersection / (union + 1e-6)
    iou[union == 0] = 1
    return iou.mean()

def calculate_bg_iou(pred, target):
    
    pred = torch.argmax(pred, dim=1)
    pred_bg = (pred == 0).float()
    target_bg = (target == 0).float()
    intersection = (pred_bg * target_bg).sum(dim=(1, 2))
    union = pred_bg.sum(dim=(1, 2)) + target_bg.sum(dim=(1, 2)) - intersection
    iou = intersection / (union + 1e-6)
    iou[union == 0] = 1
    return iou.mean()

def calculate_precision_recall(pred, target):
    
    pred = torch.argmax(pred, dim=1)
    pred_tumor = (pred == 1).float()
    target_tumor = (target == 1).float()
    true_positive = (pred_tumor * target_tumor).sum(dim=(1, 2))
    false_positive = (pred_tumor * (1 - target_tumor)).sum(dim=(1, 2))
    false_negative = ((1 - pred_tumor) * target_tumor).sum(dim=(1, 2))
    precision = true_positive / (true_positive + false_positive + 1e-6)
    recall = true_positive / (true_positive + false_negative + 1e-6)
    return precision.mean(), recall.mean()

def softmax_mse_loss(input_logits, target_logits):
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    return F.mse_loss(input_softmax, target_softmax)

def consistency_rampup(epoch, max_epochs, max_weight=0.0005):
    return max_weight * min(epoch / (max_epochs * 0.4), 1.0)

def train_model(model, ema_model, consistency_weight, consistency_criterion, alpha,
                train_loader_labeled, train_loader_unlabeled, criterion,
                optimizer, scheduler, device, epoch, num_epochs):
    model.train()
    ema_model.eval()

    running_loss = 0.0
    running_tumor_iou = 0.0
    running_f1 = 0.0
    running_bg_iou = 0.0
    running_precision = 0.0
    running_recall = 0.0

    
    labeled_data_iterator = itertools.cycle(train_loader_labeled)

    for unlabeled_batch in tqdm(train_loader_unlabeled, desc=f"Train epoch {epoch+1}/{num_epochs}"):
        labeled_batch = next(labeled_data_iterator)
        inputs_label = labeled_batch[0].to(device)
        masks_label = labeled_batch[1].to(device)

        
        min_batch_size = min(inputs_label.size(0), unlabeled_batch[0].size(0))
        inputs_label = inputs_label[:min_batch_size]
        masks_label = masks_label[:min_batch_size]
        inputs_unlabel = unlabeled_batch[0][:min_batch_size].to(device)

        student_outputs_labeled = model(inputs_label)['out']
        student_outputs_unlabeled = model(inputs_unlabel)['out']

        with torch.no_grad():
            teacher_outputs_unlabeled = ema_model(inputs_unlabel)['out']
            teacher_outputs_labeled = ema_model(inputs_label)['out']

        class_loss = criterion(student_outputs_labeled, masks_label)
        c_loss = consistency_weight * (
            consistency_criterion(student_outputs_unlabeled, teacher_outputs_unlabeled)
            + consistency_criterion(student_outputs_labeled, teacher_outputs_labeled)
        )
        loss = class_loss + c_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        
        for teacher_param, student_param in zip(ema_model.parameters(), model.parameters()):
            teacher_param.data.mul_(alpha).add_(student_param.data, alpha=1 - alpha)

        consistency_weight = consistency_rampup(epoch, num_epochs, max_weight=0.001)

   
    model.eval()
    with torch.no_grad():
        for inputs, masks in train_loader_labeled:
            inputs, masks = inputs.to(device), masks.to(device)
            outputs = ema_model(inputs)['out']

            loss = criterion(outputs, masks)
            batch_tumor_iou = calculate_iou(outputs, masks).item()
            batch_f1 = calculate_f1(outputs, masks).item()
            batch_bg_iou = calculate_bg_iou(outputs, masks).item()
            batch_precision, batch_recall = calculate_precision_recall(outputs, masks)
            batch_precision = batch_precision.item()
            batch_recall = batch_recall.item()

            running_loss += loss.item() * inputs.size(0)
            running_tumor_iou += batch_tumor_iou * inputs.size(0)
            running_f1 += batch_f1 * inputs.size(0)
            running_bg_iou += batch_bg_iou * inputs.size(0)
            running_precision += batch_precision * inputs.size(0)
            running_recall += batch_recall * inputs.size(0)

    # labeled set size
    labeled_size = len(train_loader_labeled.dataset)
    unlabeled_size = len(train_loader_unlabeled.dataset)

    epoch_loss = running_loss / labeled_size
    epoch_tumor_iou = running_tumor_iou / labeled_size
    epoch_f1 = running_f1 / labeled_size
    epoch_bg_iou = running_bg_iou / labeled_size
    epoch_precision = running_precision / labeled_size
    epoch_recall = running_recall / labeled_size
    epoch_mIoU = 0.5 * (epoch_tumor_iou + epoch_bg_iou)

    scheduler.step()  
    return epoch_loss, epoch_tumor_iou, epoch_f1, epoch_bg_iou, epoch_mIoU, epoch_precision, epoch_recall

def validate_model(model, test_loader, criterion, device):
    # ...
    model.eval()
    running_loss = 0.0
    running_tumor_iou = 0.0
    running_f1 = 0.0
    running_bg_iou = 0.0
    running_precision = 0.0
    running_recall = 0.0

    with torch.no_grad():
        for inputs, masks in tqdm(test_loader, desc="Validation Batches"):
            inputs, masks = inputs.to(device), masks.to(device)
            outputs = model(inputs)['out']

            loss = criterion(outputs, masks)
            batch_tumor_iou = calculate_iou(outputs, masks).item()
            batch_f1 = calculate_f1(outputs, masks).item()
            batch_bg_iou = calculate_bg_iou(outputs, masks).item()
            batch_precision, batch_recall = calculate_precision_recall(outputs, masks)
            batch_precision = batch_precision.item()
            batch_recall = batch_recall.item()

            running_loss += loss.item() * inputs.size(0)
            running_tumor_iou += batch_tumor_iou * inputs.size(0)
            running_f1 += batch_f1 * inputs.size(0)
            running_bg_iou += batch_bg_iou * inputs.size(0)
            running_precision += batch_precision * inputs.size(0)
            running_recall += batch_recall * inputs.size(0)

    dataset_size = len(test_loader.dataset)
    epoch_loss = running_loss / dataset_size
    epoch_tumor_iou = running_tumor_iou / dataset_size
    epoch_f1 = running_f1 / dataset_size
    epoch_bg_iou = running_bg_iou / dataset_size
    epoch_precision = running_precision / dataset_size
    epoch_recall = running_recall / dataset_size
    epoch_mIoU = 0.5*(epoch_tumor_iou + epoch_bg_iou)

    return epoch_loss, epoch_tumor_iou, epoch_f1, epoch_bg_iou, epoch_mIoU, epoch_precision, epoch_recall


def segmentation_uncertainty_score(logits):
    
    probs = F.softmax(logits, dim=1)  # [N, 2, H, W]
    pixel_entropy = - (probs * probs.log()).sum(dim=1)  # [N, H, W]
    uncertainty_img = pixel_entropy.mean(dim=(1,2))  # [N]
    return uncertainty_img  


def select_and_move_uncertain_images(model, unlabeled_folder, labeled_folder, top_ratio=0.2, batch_size=16):
    
    device = next(model.parameters()).device
    
    unlab_images = [f for f in os.listdir(unlabeled_folder) if f.endswith('.png')]
    dataset_unlab = BreastCancerSegmentationPseudo(
        root_path=unlabeled_folder,
        image_list=unlab_images,
        img_size=224,
        mask_size=224,
        cls_model='amr',
        is_augment=False
    )
    loader_unlab = torch.utils.data.DataLoader(dataset_unlab, batch_size=batch_size, shuffle=False)

    
    model.eval()
    all_uncertainties = []
    with torch.no_grad():
        idx_offset = 0
        for batch_imgs, _ in tqdm(loader_unlab, desc="[Uncertainty Calc]"):
            batch_imgs = batch_imgs.to(device)
            logits = model(batch_imgs)['out']  # [N,2,H,W]
            batch_uncert = segmentation_uncertainty_score(logits)  # shape [N]
            
            for i, unc in enumerate(batch_uncert):
                all_uncertainties.append( (idx_offset+i, unc.item()) )
            idx_offset += len(batch_imgs)

    
    all_uncertainties.sort(key=lambda x: x[1], reverse=True)
    top_count = int(len(all_uncertainties)*top_ratio)
    if top_count < 1:
        print("No images selected (top_count=0). You can adjust top_ratio.")
        return

    selected_indices = set([x[0] for x in all_uncertainties[:top_count]])

    
    print(f"[AL] total unlabeled: {len(all_uncertainties)}, select top {top_count} => move them to labeled_folder.")
    for i, filename in enumerate(unlab_images):
        if i in selected_indices:
            src_path = os.path.join(unlabeled_folder, filename)
            dst_path = os.path.join(labeled_folder, filename)
            shutil.move(src_path, dst_path)  
    print("[AL] move done.")



def main():
    
    base_path = "./BUSI/Full_Images"
    test_filename = "./BUSI_with_mask/test.xlsx"
    test_data = pd.read_excel(os.path.join(base_path, test_filename))
    images_test = test_data['Image'].tolist()
    root_path = base_path

    labeled_folder = "./pseudolabels"
    unlabeled_folder = "/unselected_images_pseudo"

    test_dataset = BreastCancerSegmentation(root_path, images_test, (224, 224), is_augment=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4, drop_last=True)

    
    model = models.segmentation.deeplabv3_resnet50(pretrained=True)
    model.classifier[4] = nn.Conv2d(256, 2, kernel_size=1)
    model = model.to(device)

    weights = torch.tensor([1.0, 2.0]).to(device)  
    criterion = nn.CrossEntropyLoss(weight=weights)

    
    ACTIVE_LEARNING_ROUNDS = 2
    EPOCHS_PER_ROUND = 10
    SELECT_RATIO = 0.3  

    
    for al_round in range(ACTIVE_LEARNING_ROUNDS):
        print(f"\n========== AL Round {al_round+1} ==========")

        
        images_train_labeled = [f for f in os.listdir(labeled_folder) if f.endswith('.png')]
        images_train_unlabeled = [f for f in os.listdir(unlabeled_folder) if f.endswith('.png')]

        train_dataset_labeled = BreastCancerSegmentationPseudo(
            labeled_folder, images_train_labeled, 224,224,
            cls_model='amr', is_augment=True
        )
        train_dataset_unlabeled = BreastCancerSegmentationPseudo(
            unlabeled_folder, images_train_unlabeled, 224,224,
            cls_model='amr', is_augment=True
        )

        train_loader_labeled = torch.utils.data.DataLoader(
            train_dataset_labeled, batch_size=16, shuffle=True, num_workers=4, drop_last=True)
        train_loader_unlabeled = torch.utils.data.DataLoader(
            train_dataset_unlabeled, batch_size=16, shuffle=True, num_workers=4, drop_last=True)

        print(f"  Labeled: {len(images_train_labeled)} images, Unlabeled: {len(images_train_unlabeled)} images")

        
        optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-2)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

        
        for epoch in range(EPOCHS_PER_ROUND):
            ema_model = copy.deepcopy(model)
            for param in ema_model.parameters():
                param.detach_()

            
            consistency_weight = 0
            consistency_criterion = softmax_mse_loss
            alpha = 0.999
            num_epochs = EPOCHS_PER_ROUND  

            train_loss, train_tumor_iou, train_f1, train_bg_iou, train_miou, train_prec, train_rec = train_model(
                model, ema_model, consistency_weight, consistency_criterion, alpha,
                train_loader_labeled, train_loader_unlabeled,
                criterion, optimizer, scheduler, device,
                epoch, EPOCHS_PER_ROUND
            )

            test_loss, test_tumor_iou, test_f1, test_bg_iou, test_miou, test_prec, test_rec = validate_model(
                ema_model, test_loader, criterion, device
            )

            print(f"[Round {al_round+1}, Epoch {epoch+1}/{EPOCHS_PER_ROUND}] "
                  f"TrainLoss={train_loss:.4f} TumIoU={train_tumor_iou:.4f} F1={train_f1:.4f} BGIoU={train_bg_iou:.4f} mIoU={train_miou:.4f} "
                  f"Prec={train_prec:.4f} Rec={train_rec:.4f} || "
                  f"TestLoss={test_loss:.4f} TumIoU={test_tumor_iou:.4f} BGIoU={test_bg_iou:.4f} mIoU={test_miou:.4f} DSC={test_f1:.4f}")

        
        if len(images_train_unlabeled) > 0:
            select_and_move_uncertain_images(
                model, unlabeled_folder, labeled_folder,
                top_ratio=SELECT_RATIO, batch_size=16
            )
        else:
            print("No more unlabeled images. Stop AL.")
            break

    
    print("\n===== Final Full Training on All labeled data =====")
    images_train_labeled = [f for f in os.listdir(labeled_folder) if f.endswith('.png')]
    train_dataset_labeled = BreastCancerSegmentationPseudo(
        labeled_folder, images_train_labeled, 224,224,
        cls_model='amr', is_augment=True
    )
    train_loader_labeled = torch.utils.data.DataLoader(
        train_dataset_labeled, batch_size=16, shuffle=True, num_workers=4, drop_last=True)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    final_epochs = 20
    for epoch in range(final_epochs):
       
        model.train()
        total_loss = 0
        total_iou = 0
        total_f1 = 0
        count = 0
        for batch_imgs, batch_masks in train_loader_labeled:
            batch_imgs = batch_imgs.to(device)
            batch_masks = batch_masks.to(device)
            logits = model(batch_imgs)['out']

            loss = criterion(logits, batch_masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            
            batch_iou = calculate_iou(logits, batch_masks).item()
            batch_f1 = calculate_f1(logits, batch_masks).item()
            bs = batch_imgs.size(0)
            total_loss += loss.item()*bs
            total_iou += batch_iou*bs
            total_f1 += batch_f1*bs
            count += bs

        scheduler.step()
        epoch_loss = total_loss / count
        epoch_iou = total_iou / count
        epoch_f1 = total_f1 / count

        
        test_loss, test_tumor_iou, test_f1, test_bg_iou, test_miou, test_prec, test_rec = validate_model(
            model, test_loader, criterion, device
        )

        print(f"[FinalTrain Epoch {epoch+1}/{final_epochs}] Loss={epoch_loss:.4f} IoU={epoch_iou:.4f} DSC={epoch_f1:.4f} || "
              f"TestLoss={test_loss:.4f} Test_IoU={test_tumor_iou:.4f} Test_DSC={test_f1:.4f}")

    
    os.makedirs("./saved_models/BrEaST/amr", exist_ok=True)
    final_model_path = "./saved_models/deeplab_pseudo_meanteacher_AL_final.pth"
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved => {final_model_path}")

if __name__ == "__main__":
    main()
