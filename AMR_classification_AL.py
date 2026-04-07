import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'

import torch
import numpy as np
import torch.utils.data as data
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F


from amr import AMRNet
from dataloader1 import BreastCancerClassification

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


seed_value = 42
torch.manual_seed(seed_value)
torch.cuda.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)
np.random.seed(seed_value)
random.seed(seed_value)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


base_path = "./BUSI"
train_filename = "train.xlsx"
test_filename = "test.xlsx"

train_data = pd.read_excel(os.path.join(base_path, train_filename))
test_data = pd.read_excel(os.path.join(base_path, test_filename))

images_train_all = train_data['Image'].tolist()
labels_train_all = train_data['Label'].tolist()
images_test = test_data['Image'].tolist()
labels_test = test_data['Label'].tolist()

root_path = "./BUSI_with_mask/Full_Images/"

INIT_LABEL_RATIO = 0.3 num_samples = len(images_train_all)
num_labeled = int(num_samples * INIT_LABEL_RATIO)


indices = list(range(num_samples))
random.shuffle(indices)
labeled_indices = indices[:num_labeled]
unlabeled_indices = indices[num_labeled:]

images_train_labeled = [images_train_all[i] for i in labeled_indices]
labels_train_labeled = [labels_train_all[i] for i in labeled_indices]

images_train_unlabeled = [images_train_all[i] for i in unlabeled_indices]


labels_train_unlabeled = [labels_train_all[i] for i in unlabeled_indices]


train_dataset_labeled = BreastCancerClassification(
    root_path,
    images_train_labeled,
    labels_train_labeled,
    224, 224,
    is_augment=True
)

train_dataset_unlabeled = BreastCancerClassification(
    root_path,
    images_train_unlabeled,
    labels_train_unlabeled,
    224, 224,
    is_augment=True
)

test_dataset = BreastCancerClassification(
    root_path,
    images_test,
    labels_test,
    224, 224,
    is_augment=False
)

def get_dataloaders():
    train_loader_labeled = torch.utils.data.DataLoader(train_dataset_labeled, batch_size=32, shuffle=True)

    train_loader_unlabeled = torch.utils.data.DataLoader(train_dataset_unlabeled, batch_size=32, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
    return train_loader_labeled, train_loader_unlabeled, test_loader

train_loader_labeled, train_loader_unlabeled, test_loader = get_dataloaders()

print(f"Initial labeled set size: {len(train_dataset_labeled)}")
print(f"Initial unlabeled set size: {len(train_dataset_unlabeled)}")
print('Test set size: ', len(test_dataset))


model = AMRNet(num_classes=2, pretrained=True).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
loss_function = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)


def train_one_epoch(model, loader, optimizer, loss_function):
    model.train()
    epoch_loss_list = []
    epoch_acc_list = []

    for (image, label) in loader:
        image, label = image.to(device), label.to(device)
        optimizer.zero_grad()
        logit_spotlight, logit_compensation = model(image)

        loss_spot = loss_function(logit_spotlight, label)
        loss_comp = loss_function(logit_compensation, label)
    
        alpha = 0.5
        loss = loss_spot + alpha * loss_comp
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            _, pred = torch.max(logit_spotlight.data, 1)
            acc = (pred == label).float().mean().item()

        epoch_loss_list.append(loss.item())
        epoch_acc_list.append(acc)

    return np.mean(epoch_loss_list), np.mean(epoch_acc_list)

def evaluate(model, loader, loss_function):
    model.eval()
    epoch_loss_list = []
    epoch_acc_list = []

    with torch.no_grad():
        for (image, label) in loader:
            image, label = image.to(device), label.to(device)
            logit_spotlight, logit_compensation = model(image)

            loss_spot = loss_function(logit_spotlight, label)
            loss_comp = loss_function(logit_compensation, label)
            loss = loss_spot + loss_comp

            _, pred = torch.max(logit_spotlight.data, 1)
            acc = (pred == label).float().mean().item()

            epoch_loss_list.append(loss.item())
            epoch_acc_list.append(acc)

    return np.mean(epoch_loss_list), np.mean(epoch_acc_list)


def compute_entropy(logits):
    probs = F.softmax(logits, dim=1)
    
    return -(probs * torch.log(probs + 1e-9)).sum(dim=1)

ACTIVE_LEARNING_ROUNDS = 1
EPOCHS_PER_ROUND = 10
SELECT_RATIO = 0.2  
for round_idx in range(ACTIVE_LEARNING_ROUNDS):
    print(f"\n========== Active Learning Round {round_idx+1} ==========")

  
    for epoch in range(EPOCHS_PER_ROUND):
        train_loss, train_acc = train_one_epoch(model, train_loader_labeled, optimizer, loss_function)
        scheduler.step()

        
        test_loss, test_acc = evaluate(model, test_loader, loss_function)

        print(f"[Epoch {epoch+1}/{EPOCHS_PER_ROUND}] "
              f"Train loss={train_loss:.4f} acc={train_acc:.4f} | "
              f"Test loss={test_loss:.4f} acc={test_acc:.4f}")

    
    model.eval()
    unlabeled_entropy = []
    with torch.no_grad():
        for idx, (image, _) in enumerate(train_loader_unlabeled):
            image = image.to(device)
            logit_spotlight, logit_compensation = model(image)
            ent = compute_entropy(logit_spotlight)  # shape = [batch_size]
           
            for b_i, e in enumerate(ent):
                absolute_idx = idx * train_loader_unlabeled.batch_size + b_i
                unlabeled_entropy.append((absolute_idx, e.item()))

  
    unlabeled_entropy.sort(key=lambda x: x[1], reverse=True)

    num_unlabeled = len(unlabeled_entropy)
    select_num = int(num_unlabeled * SELECT_RATIO)
    selected_indices = unlabeled_entropy[:select_num]
    print(f"Unlabeled Data: {num_unlabeled}, select top {select_num} uncertain samples => move to labeled set.")


    selected_indices_set = set([x[0] for x in selected_indices])

    new_unlabeled_images = []
    new_unlabeled_labels = []

    for i in range(num_unlabeled):
        img_name = images_train_unlabeled[i]
     
        label_u = labels_train_unlabeled[i]
        if i in selected_indices_set:
            train_dataset_labeled.image_list.append(img_name)
            train_dataset_labeled.label_list.append(label_u)
        else:
            new_unlabeled_images.append(img_name)
            new_unlabeled_labels.append(label_u)


    images_train_unlabeled = new_unlabeled_images
    labels_train_unlabeled = new_unlabeled_labels
    train_dataset_unlabeled.image_list = images_train_unlabeled
    train_dataset_unlabeled.label_list = labels_train_unlabeled

 
    train_loader_labeled, train_loader_unlabeled, test_loader = get_dataloaders()

print("\n========== Final Full Training (All Labeled) ==========")
for epoch in range(10):
    train_loss, train_acc = train_one_epoch(model, train_loader_labeled, optimizer, loss_function)
    scheduler.step()
    test_loss, test_acc = evaluate(model, test_loader, loss_function)
    print(f"[Final Epoch {epoch+1}/10] "
          f"Train loss={train_loss:.4f} acc={train_acc:.4f} | "
          f"Test loss={test_loss:.4f} acc={test_acc:.4f}")


save_path = "./AMRNet_weights_AL3.pth"
torch.save(model.state_dict(), save_path)
print(f"Active Learning 3 Rounds Completed. Model Saved => {save_path}")


model.eval()
all_labels = []
all_preds = []
all_probs = []

with torch.no_grad():
    for image, label in test_loader:
        image = image.to(device)
        label = label.to(device)
        logit_spotlight, logit_compensation = model(image)
        probs_spot = F.softmax(logit_spotlight, dim=1)
        preds_spot = torch.argmax(probs_spot, dim=1)

        all_labels.extend(label.cpu().numpy())
        all_preds.extend(preds_spot.cpu().numpy())
        all_probs.extend(probs_spot[:,1].cpu().numpy())  

all_labels = np.array(all_labels)
all_preds = np.array(all_preds)
all_probs = np.array(all_probs)

acc = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds)
recall = recall_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)

print("\nFinal Test Metrics After AL:")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")

fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
roc_auc = auc(fpr, tpr)
print(f"ROC AUC:   {roc_auc:.4f}")

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0,1],[0,1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()
