import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import sys

import torch
import torchvision
from torchvision import datasets, models
from torchvision.transforms import functional as FT
from torchvision import transforms as T
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, sampler, random_split, Dataset
import copy
import math
from PIL import Image
import cv2
import albumentations as A  # our data augmentation library

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")
from collections import defaultdict, deque
import datetime
import time
from tqdm import tqdm # progress bar
from torchvision.utils import draw_bounding_boxes

print(torch.__version__)
print(torchvision.__version__)

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json

from albumentations.pytorch import ToTensorV2
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchmetrics.detection.mean_ap import MeanAveragePrecision

# Add variables to store metrics
all_losses = []
all_losses_dict = []
val_losses = []
val_losses_dict = []

# Add variables for MAP, Precision, Recall
train_map = []
train_precision = []
train_recall = []
val_map = []
val_precision = []
val_recall = []

def calculate_metrics(predictions, targets, coco_gt):
    # Convert predictions to COCO format
    coco_pred = []
    for pred in predictions:
        image_id = pred['image_id'].item()
        boxes = pred['boxes'].cpu().numpy()
        scores = pred['scores'].cpu().numpy()
        labels = pred['labels'].cpu().numpy()
        
        for box, score, label in zip(boxes, scores, labels):
            coco_pred.append({
                'image_id': image_id,
                'category_id': label,
                'bbox': [box[0], box[1], box[2]-box[0], box[3]-box[1]],  # Convert to [x,y,w,h]
                'score': score
            })
    
    # Save predictions to temporary file
    with open('temp_pred.json', 'w') as f:
        json.dump(coco_pred, f)
    
    # Load predictions
    coco_pred = coco_gt.loadRes('temp_pred.json')
    
    # Initialize COCO eval
    coco_eval = COCOeval(coco_gt, coco_pred, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    # Get metrics
    map_score = coco_eval.stats[0]  # AP at IoU=0.5:0.95
    precision = coco_eval.stats[1]  # AP at IoU=0.5
    recall = coco_eval.stats[8]     # AR at IoU=0.5:0.95
    
    # Clean up temporary file
    os.remove('temp_pred.json')
    
    return map_score, precision, recall

def get_transforms(train=False):
    if train:
        transform = A.Compose([
            A.Resize(600, 600), # our input size can be 600px
            A.HorizontalFlip(p=0.3),
            A.VerticalFlip(p=0.3),
            A.RandomBrightnessContrast(p=0.1),
            A.ColorJitter(p=0.1),
            A.HorizontalFlip(p=0.2),
            A.VerticalFlip(p=0.1),

            ToTensorV2()
        ], bbox_params=A.BboxParams(format='coco'))
    else:
        transform = A.Compose([
            A.Resize(600, 600), # our input size can be 600px
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='coco'))
    return transform

class LoadData(datasets.VisionDataset):
    def __init__(self, root, split='train', transform=None, target_transform=None, transforms=None):
        # the 3 transform parameters are reuqired for datasets.VisionDataset
        super().__init__(root, transforms, transform, target_transform)
        self.split = split #train, valid, test
        self.coco = COCO(os.path.join(root, split, "_annotations.coco.json")) # annotatiosn stored here
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.ids = [id for id in self.ids if (len(self._load_target(id)) > 0)]

    def _load_image(self, id: int):
        path = self.coco.loadImgs(id)[0]['file_name']
        image = cv2.imread(os.path.join(self.root, self.split, path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    def _load_target(self, id):
        return self.coco.loadAnns(self.coco.getAnnIds(id))

    def __getitem__(self, index):
        id = self.ids[index]
        image = self._load_image(id)
        target = self._load_target(id)
        target = copy.deepcopy(self._load_target(id))

        boxes = [t['bbox'] + [t['category_id']] for t in target] # required annotation format for albumentations
        if self.transforms is not None:
            transformed = self.transforms(image=image, bboxes=boxes)

        image = transformed['image']
        boxes = transformed['bboxes']

        new_boxes = [] # convert from xywh to xyxy
        for box in boxes:
            xmin = box[0]
            xmax = xmin + box[2]
            ymin = box[1]
            ymax = ymin + box[3]
            new_boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.tensor(new_boxes, dtype=torch.float32)

        targ = {} # here is our transformed target
        targ['boxes'] = boxes
        targ['labels'] = torch.tensor([t['category_id'] for t in target], dtype=torch.int64)
        targ['image_id'] = torch.tensor([t['image_id'] for t in target])
        targ['area'] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) # we have a different area
        targ['iscrowd'] = torch.tensor([t['iscrowd'] for t in target], dtype=torch.int64)
        return image.div(255), targ # scale images
    def __len__(self):
        return len(self.ids)

# === TRAIN ONE EPOCH ===
def train_one_epoch(model, optimizer, loader, device, epoch, coco_gt):
    model.to(device)
    model.train()

    epoch_losses = []
    epoch_losses_dict = []
    all_predictions = []
    all_targets = []

    for images, targets in tqdm(loader, desc=f"Epoch {epoch} [Train]"):
        images = list(image.to(device) for image in images)
        targets = [{k: torch.tensor(v).to(device) for k, v in t.items()} for t in targets]

        # Forward pass
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_dict_item = {k: v.item() for k, v in loss_dict.items()}
        loss_value = losses.item()

        epoch_losses.append(loss_value)
        epoch_losses_dict.append(loss_dict_item)

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        # Get predictions for metrics
        with torch.no_grad():
            model.eval()  # Temporarily set to eval mode
            predictions = model(images)
            model.train()  # Set back to train mode
            all_predictions.extend(predictions)
            all_targets.extend(targets)

    # Calculate metrics
    map_score, precision, recall = calculate_metrics(all_predictions, all_targets, coco_gt)
    train_map.append(map_score)
    train_precision.append(precision)
    train_recall.append(recall)

    # Epoch summary
    epoch_loss_mean = np.mean(epoch_losses)
    epoch_loss_dict_mean = pd.DataFrame(epoch_losses_dict).mean().to_dict()

    all_losses.append(epoch_loss_mean)
    all_losses_dict.append(epoch_loss_dict_mean)

    print("✅ Train Epoch {}, lr: {:.6f}, loss: {:.6f}, loss_classifier: {:.6f}, loss_box: {:.6f}, loss_rpn_box: {:.6f}, loss_object: {:.6f}, mAP: {:.4f}, Precision: {:.4f}, Recall: {:.4f}".format(
        epoch, optimizer.param_groups[0]['lr'], epoch_loss_mean,
        epoch_loss_dict_mean['loss_classifier'],
        epoch_loss_dict_mean['loss_box_reg'],
        epoch_loss_dict_mean['loss_rpn_box_reg'],
        epoch_loss_dict_mean['loss_objectness'],
        map_score, precision, recall
    ))

# === VALIDATION ===
def evaluate_one_epoch(model, loader, device, epoch, coco_gt):
    model.to(device)
    model.eval()  # Set to eval mode for validation

    epoch_losses = []
    epoch_losses_dict = []
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for images, targets in tqdm(loader, desc=f"Epoch {epoch} [Val]"):
            images = list(image.to(device) for image in images)
            targets = [{k: torch.tensor(v).to(device) for k, v in t.items()} for t in targets]

            # Get predictions
            predictions = model(images)
            all_predictions.extend(predictions)
            all_targets.extend(targets)

            # Calculate loss
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            loss_dict_item = {k: v.item() for k, v in loss_dict.items()}
            epoch_losses.append(losses.item())
            epoch_losses_dict.append(loss_dict_item)

    # Calculate metrics
    map_score, precision, recall = calculate_metrics(all_predictions, all_targets, coco_gt)
    val_map.append(map_score)
    val_precision.append(precision)
    val_recall.append(recall)

    val_loss_mean = np.mean(epoch_losses)
    val_loss_dict_mean = pd.DataFrame(epoch_losses_dict).mean().to_dict()

    val_losses.append(val_loss_mean)
    val_losses_dict.append(val_loss_dict_mean)

    print("✅ Validation Epoch {}, loss: {:.6f}, loss_classifier: {:.6f}, loss_box: {:.6f}, loss_rpn_box: {:.6f}, loss_object: {:.6f}, mAP: {:.4f}, Precision: {:.4f}, Recall: {:.4f}".format(
        epoch, val_loss_mean,
        val_loss_dict_mean['loss_classifier'],
        val_loss_dict_mean['loss_box_reg'],
        val_loss_dict_mean['loss_rpn_box_reg'],
        val_loss_dict_mean['loss_objectness'],
        map_score, precision, recall
    ))

# === COLLATE ===
def collate_fn(batch):
    return tuple(zip(*batch))

# === VISUALIZATION ===
def Loss_Curve(all_losses=all_losses, val_losses=val_losses):
    plt.plot(range(1, len(all_losses)+1), all_losses, marker='o', label='Train Loss')
    plt.plot(range(1, len(val_losses)+1), val_losses, marker='s', label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve (Train vs Validation)')
    plt.grid(True)
    plt.legend()
    plt.show()

def visualize_losses_large(all_losses_dict):
    all_losses_dict = pd.DataFrame(all_losses_dict)
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
    axes = axes.flatten()
    loss_types = ['loss_classifier', 'loss_objectness', 'loss_box_reg', 'loss_rpn_box_reg']
    titles = ['Loss Classifier', 'Loss Objectness', 'Loss Box', 'Loss RPN Box']
    for i, loss_type in enumerate(loss_types):
        ax = axes[i]
        ax.plot(range(1, len(all_losses_dict)+1), all_losses_dict[loss_type], marker='o', label=loss_type)
        ax.set_title(titles[i])
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True)
    plt.tight_layout()
    plt.show()

def visualize_val_losses_large(val_losses_dict):
    val_losses_dict = pd.DataFrame(val_losses_dict)
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
    axes = axes.flatten()
    loss_types = ['loss_classifier', 'loss_objectness', 'loss_box_reg', 'loss_rpn_box_reg']
    titles = ['Val Loss Classifier', 'Val Loss Objectness', 'Val Loss Box', 'Val Loss RPN Box']
    for i, loss_type in enumerate(loss_types):
        ax = axes[i]
        ax.plot(range(1, len(val_losses_dict)+1), val_losses_dict[loss_type], marker='o', label=loss_type)
        ax.set_title(titles[i])
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True)
    plt.tight_layout()
    plt.show()

def visualize_metrics():
    # Create figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))
    
    # Plot mAP
    ax1.plot(range(1, len(train_map)+1), train_map, marker='o', label='Train mAP')
    ax1.plot(range(1, len(val_map)+1), val_map, marker='s', label='Val mAP')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('mAP')
    ax1.set_title('mAP Curve (Train vs Validation)')
    ax1.grid(True)
    ax1.legend()
    
    # Plot Precision
    ax2.plot(range(1, len(train_precision)+1), train_precision, marker='o', label='Train Precision')
    ax2.plot(range(1, len(val_precision)+1), val_precision, marker='s', label='Val Precision')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision Curve (Train vs Validation)')
    ax2.grid(True)
    ax2.legend()
    
    # Plot Recall
    ax3.plot(range(1, len(train_recall)+1), train_recall, marker='o', label='Train Recall')
    ax3.plot(range(1, len(val_recall)+1), val_recall, marker='s', label='Val Recall')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Recall')
    ax3.set_title('Recall Curve (Train vs Validation)')
    ax3.grid(True)
    ax3.legend()
    
    plt.tight_layout()
    plt.show()

# === MAIN ===
def main():
    dataset_path = "/content/drive/MyDrive/CD2/Dataset2"
    coco = COCO(os.path.join(dataset_path, "valid", "_annotations.coco.json"))
    categories = coco.cats
    n_classes = len(categories.keys()) + 1
    print(categories)

    classes = [i[1]['name'] for i in categories.items()]
    print(classes)

    train_dataset = LoadData(root=dataset_path, transforms=get_transforms(True))
    val_dataset = LoadData(root=dataset_path, split='test', transforms=get_transforms(False))

    model = models.detection.fasterrcnn_resnet50_fpn_v2(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(in_features, n_classes)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0, collate_fn=collate_fn)

    device = torch.device("cuda")
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.01, momentum=0.9, nesterov=True, weight_decay=1e-4)

    num_epochs = 10
    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, train_loader, device, epoch, coco)
        evaluate_one_epoch(model, val_loader, device, epoch, coco)

    torch.save(model.state_dict(), 'model_vehicle.pth')

    # Visualize all metrics
    Loss_Curve()
    visualize_losses_large(all_losses_dict)
    visualize_val_losses_large(val_losses_dict)
    visualize_metrics()

if __name__ == '__main__':
    main()