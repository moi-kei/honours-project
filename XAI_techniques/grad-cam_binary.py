import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_custom_model(model_path):
    model = models.resnet50()
    model.fc = nn.Sequential(
        nn.Linear(2048, 256),
        nn.ReLU(),
        nn.Dropout(),
        nn.Linear(256, 1)
    )
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

#model_path = "models/brain_tumour_pretrained.pth"
model_path = "models/brain_tumour.pth"
class_names = ["no_tumour", "tumour"]
model = load_custom_model(model_path)

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

def load_image(img_path):
    img = Image.open(img_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean.tolist(), std=std.tolist())
    ])
    img_tensor = transform(img).unsqueeze(0).to(device)
    return img, img_tensor, np.array(img.resize((224, 224)))

def generate_grad_cam(model, img_tensor):
    model.eval()
    gradients = []
    activations = []

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def forward_hook(module, input, output):
        activations.append(output)

    target_layer = model.layer4[-1].conv3
    target_layer.register_forward_hook(forward_hook)
    target_layer.register_backward_hook(backward_hook)

    output = model(img_tensor)
    model.zero_grad()
    output_sigmoid = torch.sigmoid(output)
    output_sigmoid.backward(torch.ones_like(output_sigmoid))

    grad = gradients[0].cpu().data.numpy()
    act = activations[0].cpu().data.numpy()

    weights = np.mean(grad, axis=(2, 3), keepdims=True)
    cam = np.sum(weights * act, axis=1).squeeze()
    cam = np.maximum(cam, 0)
    cam = cam - np.min(cam)
    if np.max(cam) != 0:
        cam = cam / np.max(cam)
    else:
        cam = np.zeros_like(cam)
    return cam

def overlay_heatmap(image, heatmap):
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlayed = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)
    return overlayed

image_dir = 'test_images/brain_tumour_test_images'
#output_dir = "results/brain_tumour/pretrained/grad-cam"
output_dir = "results/brain_tumour/untrained/grad-cam"
os.makedirs(output_dir, exist_ok=True)

for img_name in os.listdir(image_dir):
    img_path = os.path.join(image_dir, img_name)
    if not os.path.isfile(img_path):
        continue

    original_image, img_tensor, raw_image = load_image(img_path)

    with torch.no_grad():
        output = model(img_tensor)
        prob = torch.sigmoid(output).item()
    pred_class = 1 if prob > 0.5 else 0
    print(f'Predicted for {img_name}:', class_names[pred_class])

    start_time = time.time()
    heatmap = generate_grad_cam(model, img_tensor)
    end_time = time.time()

    gradcam_result = overlay_heatmap(raw_image, heatmap)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(raw_image)
    axes[0].axis("off")
    axes[0].set_title("Original Image")
    axes[1].imshow(gradcam_result)
    axes[1].axis("off")
    axes[1].set_title(f"{class_names[pred_class]} ({prob:.2f})")

    print(f"Explanation time for {img_name}: {end_time - start_time:.2f} seconds")

    save_path = os.path.join(output_dir, f"{os.path.splitext(img_name)[0]}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()