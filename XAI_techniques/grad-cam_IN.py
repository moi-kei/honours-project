import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import models, transforms
from PIL import Image
import cv2
import json
import urllib.request
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).to(device)
model.eval()

url = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
with urllib.request.urlopen(url) as response:
    class_names = [v[1] for v in json.load(response).values()]

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

def generate_grad_cam(model, img_tensor, target_class):
    model.eval()
    gradients = []
    activations = []
    def hook_function(module, grad_input, grad_output):
        gradients.append(grad_output[0])
    def forward_hook(module, input, output):
        activations.append(output)
    target_layer = model.layer4[-1].conv3
    target_layer.register_forward_hook(forward_hook)
    target_layer.register_backward_hook(hook_function)
    output = model(img_tensor)
    model.zero_grad()
    class_score = output[0, target_class]
    class_score.backward()
    grad = gradients[0].cpu().data.numpy()
    act = activations[0].cpu().data.numpy()
    weights = np.mean(grad, axis=(2, 3), keepdims=True)
    cam = np.sum(weights * act, axis=1).squeeze()
    cam = np.maximum(cam, 0)
    cam = cam - np.min(cam)
    cam = cam / np.max(cam)
    return cam

def overlay_heatmap(image, heatmap):
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlayed = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)
    return overlayed

image_dir = 'imageNet_test_images'
os.makedirs("gradcam_results", exist_ok=True)
#image_dir = 'consistency_test'
#os.makedirs("gradcam_consistency", exist_ok=True)

for img_name in os.listdir(image_dir):
    img_path = os.path.join(image_dir, img_name)
    if not os.path.isfile(img_path):
        continue
    original_image, img_tensor, raw_image = load_image(img_path)
    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()[0]
    top5_indices = np.argsort(probs)[-5:][::-1]
    top5_classes = [class_names[i] for i in top5_indices]
    top5_probs = [probs[i] for i in top5_indices]
    print(f'Predicted for {img_name}: {top5_classes[0]} ({top5_probs[0]:.2f})')
    fig, axes = plt.subplots(1, 6, figsize=(18, 5))
    axes[0].imshow(raw_image)
    axes[0].axis("off")
    axes[0].set_title("Original")

    start_time = time.time()

    for i, class_idx in enumerate(top5_indices):
        heatmap = generate_grad_cam(model, img_tensor, class_idx)
        gradcam_result = overlay_heatmap(raw_image, heatmap)
        axes[i + 1].imshow(gradcam_result)
        axes[i + 1].axis("off")
        axes[i + 1].set_title(f"{top5_classes[i]}\n{top5_probs[i]:.2f}")

    end_time = time.time()
    duration = end_time - start_time
    print(f"Explanation time for {img_name}: {duration:.2f} seconds")

    save_path = os.path.join("gradcam_results", f"{os.path.splitext(img_name)[0]}.png")
    #save_path = os.path.join("gradcam_consistency", f"{os.path.splitext(img_name)[0]}.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()