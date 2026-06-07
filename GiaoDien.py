import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import os
import torch
from torchvision import models
from torchvision.utils import draw_bounding_boxes
from pycocotools.coco import COCO
import albumentations as A
from albumentations.pytorch import ToTensorV2
import warnings
warnings.filterwarnings("ignore")

selected_image_path = None
original_image = None

def get_transforms(train=False):
    if train:
        transform = A.Compose([
            A.Resize(600, 600),
            A.HorizontalFlip(p=0.3),
            A.VerticalFlip(p=0.3),
            A.RandomBrightnessContrast(p=0.1),
            A.ColorJitter(p=0.1),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='coco'))
    else:
        transform = A.Compose([
            A.Resize(600, 600),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='coco'))
    return transform

def load_image():
    global selected_image_path, original_image
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.webp")])
    if file_path:
        selected_image_path = file_path
        image_pil = Image.open(file_path).convert("RGB")
        original_image = image_pil.copy()
        display_img = image_pil.resize((600, 400))
        tk_img = ImageTk.PhotoImage(display_img)
        original_label.config(image=tk_img)
        original_label.image = tk_img
        result_label.config(text="Image loaded. Click 'Predict' to run detection.")

def predict_image():
    global selected_image_path, original_image
    if not selected_image_path:
        result_label.config(text="Please load an image first!")
        return

    image_np = np.array(original_image)
    transform = get_transforms(False)
    transformed = transform(image=image_np, bboxes=[])
    image_tensor = transformed['image'].float().div(255).unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = model(image_tensor)

    pred = prediction[0]
    score_filter = pred['scores'] > 0.7
    pred = {k: v[score_filter].to('cpu') for k, v in pred.items()}

    counts = {}
    for label in pred['labels']:
        name = id2name.get(label.item(), "Unknown")
        counts[name] = counts.get(name, 0) + 1

    count_text = "Detected Vehicles:\n" + "\n".join(f"{k}: {v}" for k, v in counts.items())

    drawn_image = draw_bounding_boxes(
        image_tensor[0].mul(255).byte().cpu(),
        boxes=pred['boxes'].cpu(),
        labels=[id2name.get(i.item(), "Unknown") for i in pred['labels']],
        colors="red",
        width=3,
        font_size=20
    ).permute(1, 2, 0).numpy()

    display_image = Image.fromarray(drawn_image).resize((600, 400))
    tk_image = ImageTk.PhotoImage(display_image)

    result_label.config(text=count_text)
    result_label.update()
    predicted_label.config(image=tk_image)
    predicted_label.image = tk_image

# Load COCO labels
dataset_path = "Dataset"
coco = COCO(os.path.join(dataset_path, "train", "_annotations.coco.json"))
categories = coco.cats
id2name = {cat_id: cat_info['name'] for cat_id, cat_info in categories.items()}
n_classes = len(id2name) + 1

# Load model
model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(in_features, n_classes)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
model.load_state_dict(torch.load('model_test_final_2.pth', map_location=device))
model.eval()

# ------------------ GUI ---------------------
root = tk.Tk()
root.title("Vehicle Detection")
root.attributes('-fullscreen', True)

top_frame = tk.Frame(root, bg="#222")
top_frame.pack(fill=tk.X)

title_label = tk.Label(top_frame, text="Vehicle Detection System", font=("Helvetica", 28), bg="#222", fg="white", pady=10)
title_label.pack()

button_frame = tk.Frame(root, pady=10)
button_frame.pack()

btn_load = tk.Button(button_frame, text="Load Image", font=("Helvetica", 16), command=load_image, bg="#1976D2", fg="white", padx=20, pady=10)
btn_load.grid(row=0, column=0, padx=20)

btn_predict = tk.Button(button_frame, text="Predict", font=("Helvetica", 16), command=predict_image, bg="#388E3C", fg="white", padx=20, pady=10)
btn_predict.grid(row=0, column=1, padx=20)

btn_exit = tk.Button(button_frame, text="Exit", font=("Helvetica", 16), command=root.destroy, bg="#D32F2F", fg="white", padx=20, pady=10)
btn_exit.grid(row=0, column=2, padx=20)

image_frame = tk.Frame(root)
image_frame.pack(pady=20)

original_label = tk.Label(image_frame, text="")
original_label.grid(row=0, column=0, padx=20)

predicted_label = tk.Label(image_frame, text="")
predicted_label.grid(row=0, column=1, padx=20)

result_label = tk.Label(root, text="", font=("Helvetica", 16), bg="#f0f0f0", justify=tk.LEFT)
result_label.pack(pady=20)

root.mainloop()
