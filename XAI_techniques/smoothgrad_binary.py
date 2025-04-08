import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
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
model = load_custom_model(model_path)
class_names = ["no_tumour", "tumour"]

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

def generate_smoothgrad(model, img_tensor, target_score, num_samples=50, noise_level=0.2):
    model.eval()
    img_tensor.requires_grad_()
    smooth_grad = torch.zeros_like(img_tensor, device=device)
    for _ in range(num_samples):
        noise = torch.randn_like(img_tensor) * noise_level
        noisy_img = img_tensor + noise
        noisy_img.requires_grad_()
        noisy_img.retain_grad()
        output = model(noisy_img)
        model.zero_grad()
        score = target_score(output)
        score.backward()
        if noisy_img.grad is not None:
            smooth_grad += noisy_img.grad.abs()
    smooth_grad /= num_samples
    saliency = smooth_grad.squeeze().detach().cpu().numpy()
    saliency = np.max(saliency, axis=0)
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
    saliency = 1.0 - saliency
    return saliency

image_dir = "test_images/brain_tumour_test_images"
#output_dir = "results/brain_tumour/pretrained/smoothgrad"
output_dir = "results/brain_tumour/untrained/smoothgrad"
os.makedirs(output_dir, exist_ok=True)

for img_name in os.listdir(image_dir):
    img_path = os.path.join(image_dir, img_name)
    if not os.path.isfile(img_path):
        continue
    original_image, img_tensor, raw_image = load_image(img_path)
    with torch.no_grad():
        output = model(img_tensor)
        prob = torch.sigmoid(output)[0][0].item()
        pred_class = 1 if prob > 0.5 else 0
        label = class_names[pred_class]
        confidence = prob if pred_class == 1 else 1 - prob
        print(f'Predicted for {img_name}: {label} ({confidence:.2f})')
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(raw_image)
    axes[0].axis("off")
    axes[0].set_title("Original Image")
    start_time = time.time()
    model.zero_grad()
    target_score = lambda o: torch.sigmoid(o)[0][0] if pred_class == 1 else 1 - torch.sigmoid(o)[0][0]
    saliency = generate_smoothgrad(model, img_tensor, target_score)
    axes[1].imshow(saliency, cmap="gray")
    axes[1].axis("off")
    axes[1].set_title(f"{label}\n{confidence:.2f}")
    end_time = time.time()
    print(f"Explanation time for {img_name}: {end_time - start_time:.2f} seconds")
    save_path = os.path.join(output_dir, f"{os.path.splitext(img_name)[0]}.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()