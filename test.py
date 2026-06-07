# -*- coding: utf-8 -*-

from torchmetrics.detection.mean_ap import MeanAveragePrecision
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import sys
import random
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
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")
from collections import defaultdict, deque, Counter
import datetime
import time
from tqdm import tqdm # progress bar
from torchvision.utils import draw_bounding_boxes

from pycocotools.coco import COCO

from albumentations.pytorch import ToTensorV2

def get_transforms(train=False):
    if train:
        transform = A.Compose([
            A.Resize(600, 600), # our input size can be 600px
            A.HorizontalFlip(p=0.3),
            A.VerticalFlip(p=0.3),
            A.RandomBrightnessContrast(p=0.1),
            A.ColorJitter(p=0.1),
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

def train_one_epoch(model, optimizer, loader, device, epoch):
    model.to(device)
    model.train()

#     lr_scheduler = None
#     if epoch == 0:
#         warmup_factor = 1.0 / 1000 # do lr warmup
#         warmup_iters = min(1000, len(loader) - 1)

#         lr_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor = warmup_factor, total_iters=warmup_iters)

    all_losses = []
    all_losses_dict = []

    for images, targets in tqdm(loader):
        images = list(image.to(device) for image in images)
        targets = [{k: torch.tensor(v).to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets) # the model computes the loss automatically if we pass in targets
        losses = sum(loss for loss in loss_dict.values())
        loss_dict_append = {k: v.item() for k, v in loss_dict.items()}
        loss_value = losses.item()

        all_losses.append(loss_value)
        all_losses_dict.append(loss_dict_append)

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping trainig") # train if loss becomes infinity
            print(loss_dict)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

#         if lr_scheduler is not None:
#             lr_scheduler.step() #

    all_losses_dict = pd.DataFrame(all_losses_dict) # for printing
    print("Epoch {}, lr: {:.6f}, loss: {:.6f}, loss_classifier: {:.6f}, loss_box: {:.6f}, loss_rpn_box: {:.6f}, loss_object: {:.6f}".format(
        epoch, optimizer.param_groups[0]['lr'], np.mean(all_losses),
        all_losses_dict['loss_classifier'].mean(),
        all_losses_dict['loss_box_reg'].mean(),
        all_losses_dict['loss_rpn_box_reg'].mean(),
        all_losses_dict['loss_objectness'].mean()
    ))

def collate_fn(batch):
    return tuple(zip(*batch))

def predict_image(image_path):
    # Load the image
    image_pil = Image.open(image_path).convert("RGB")
    image_np = np.array(image_pil)

    # Apply the same transformations as for the training images
    transform = get_transforms(False)
    transformed = transform(image=image_np, bboxes=[])
    image = transformed['image']

    # Convert the image to a floating point tensor and normalize it
    image = image.float().div(255)

    # Add an extra dimension as the model expects a batch
    image_batch = image.unsqueeze(0).to(device)

    # Make a prediction
    with torch.no_grad():
        prediction = model(image_batch)

    # Filter out predictions with a confidence score less than 0.5
    pred = prediction[0]
    high_score_idx = pred['scores'] > 0.7
    pred = {k: v[high_score_idx] for k, v in pred.items()}

    # Get labels
    labels = [classes[i] for i in pred['labels'].cpu().tolist()]

    # Đếm số lượng từng loại phương tiện
    label_counts = Counter(labels)

    # Vẽ bounding boxes
    img_with_boxes = draw_bounding_boxes(
        (image * 255).byte().cpu(),
        pred['boxes'].cpu(),
        labels,
        width=4
    )

    # Chuyển ảnh sang định dạng numpy để dùng OpenCV ghi text
    img_np = img_with_boxes.permute(1, 2, 0).numpy()
    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # Viết số lượng từng loại phương tiện lên góc trái
    y_offset = 30
    for label, count in label_counts.items():
        text = f"{label}: {count}"
        cv2.putText(img_np, text, (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        y_offset += 35

    # Chuyển lại sang RGB để hiển thị bằng matplotlib
    img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

    # Hiển thị kết quả
    plt.figure(figsize=(14, 10))
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.show()

from collections import Counter

def test_video():
    cap = cv2.VideoCapture('/content/drive/MyDrive/video-deeplearning/test.mp4')

    # Xác định codec và tạo đối tượng VideoWriter để ghi video kết quả
    codec = cv2.VideoWriter_fourcc(*'XVID')
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_video = cv2.VideoWriter('output_video_test.mp4', codec, 30, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Chuyển frame từ BGR (OpenCV) sang RGB (PyTorch)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_tensor = T.ToTensor()(frame_rgb).unsqueeze(0).to(device)

        with torch.no_grad():
            prediction = model(img_tensor)

        pred = prediction[0]
        scores = pred['scores']
        keep = scores > 0.9
        boxes = pred['boxes'][keep]
        labels = pred['labels'][keep]

        label_names = [classes[label.item()] for label in labels]
        label_counts = Counter(label_names)

        # Vẽ bounding boxes và nhãn
        for box, label in zip(boxes, label_names):
            box = box.detach().cpu().numpy().astype(int)
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            cv2.putText(frame, label, (box[0], box[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Ghi số lượng từng loại phương tiện lên góc trái frame
        y_offset = 30
        for label, count in label_counts.items():
            text = f"{label}: {count}"
            cv2.putText(frame, text, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            y_offset += 30

        # Ghi frame đã xử lý vào video kết quả
        output_video.write(frame)

    cap.release()
    output_video.release()
    cv2.destroyAllWindows()
def evaluate_model(model, test_loader, device):
    model.eval()
    
    # Khởi tạo metric mAP
    metric_map = MeanAveragePrecision()
    
    all_predictions = []
    all_targets = []

    # Tắt gradient trong quá trình đánh giá
    with torch.no_grad():
        for images, targets in tqdm(test_loader, desc="Evaluating"):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            outputs = model(images)

            for output, target in zip(outputs, targets):
                predictions = {
                    "boxes": output["boxes"].cpu(),
                    "scores": output["scores"].cpu(),
                    "labels": output["labels"].cpu(),
                }
                ground_truth = {
                    "boxes": target["boxes"].cpu(),
                    "labels": target["labels"].cpu(),
                }
                all_predictions.append(predictions)
                all_targets.append(ground_truth)

    # Tính toán chỉ số
    metric_map.update(all_predictions, all_targets)
    results = metric_map.compute()

    # Trích xuất các chỉ số chính
    map_val = float(f"{results['map'].item():.4f}")
    map_50_val = float(f"{results['map_50'].item():.4f}")
    map_75_val = float(f"{results['map_75'].item():.4f}")

    # Tạo bảng kết quả
    data = {
        "Metric": ["mAP (IoU=0.5:0.95)", "Precision (IoU=0.5)", "Recall (IoU=0.75)"],
        "Score": [map_val, map_50_val, map_75_val]
    }
    df = pd.DataFrame(data)

    # Hiển thị bảng bằng matplotlib
    fig, ax = plt.subplots(figsize=(6, 2))
    ax.axis('off')
    table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)
    plt.title("Detection Performance Report", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

    return df, results
dataset_path = "Dataset"
coco = COCO(os.path.join(dataset_path, "train", "_annotations.coco.json"))
categories = coco.cats
n_classes = len(categories.keys()) + 1
print(categories)

classes = [i[1]['name'] for i in categories.items()]
print(classes)

model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(in_features, n_classes)


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

model.load_state_dict(torch.load('model_test_final_2.pth', map_location=device))
model.eval()

test_dataset = LoadData(root=dataset_path, split="test", transforms=get_transforms(False))
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_fn)
# num_images_to_display = 5
# num_images_displayed = 0

# for i in range(len(test_dataset)):
#     # Lấy ảnh từ tập dữ liệu thử nghiệm
#     img, _ = test_dataset[i]  # img là tensor (C, H, W)
#     img_int = torch.tensor(img * 255, dtype=torch.uint8)

#     # Dự đoán với mô hình
#     with torch.no_grad():
#         prediction = model([img.to(device)])
#         pred = prediction[0]

#     # Lọc các prediction có độ chắc chắn > 0.8
#     score_thresh = 0.8
#     keep = pred['scores'] > score_thresh
#     boxes = pred['boxes'][keep]
#     labels = pred['labels'][keep]

#     # Đếm số lượng từng loại phương tiện
#     label_names = [classes[label.item()] for label in labels]
#     label_counts = Counter(label_names)

#     # Vẽ bounding boxes
#     img_with_boxes = draw_bounding_boxes(
#         img_int.cpu(),
#         boxes.cpu(),
#         label_names,
#         colors="lime",
#         width=4
#     )

#     # Chuyển sang định dạng để hiển thị bằng matplotlib (C, H, W) -> (H, W, C)
#     img_np = img_with_boxes.permute(1, 2, 0).numpy()

#     # Vẽ lên bằng matplotlib và thêm text thống kê
#     fig, ax = plt.subplots(figsize=(14, 10))
#     ax.imshow(img_np)
#     y_offset = 20
#     for label, count in label_counts.items():
#         ax.text(10, y_offset, f"{label}: {count}", fontsize=14,
#                 color="red", backgroundcolor="white")
#         y_offset += 25

#     plt.axis('off')
#     plt.show()

#     num_images_displayed += 1
#     if num_images_displayed >= num_images_to_display:
#         break
# predict_image('4.jpg')
# test_video()
# test_dataset = LoadData(root=dataset_path, split="test", transforms=get_transforms(False))
# df_result, raw_metrics = evaluate_model(model, test_loader, device)



