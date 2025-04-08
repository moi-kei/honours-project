import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import models, transforms
from PIL import Image
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

def generate_smoothgrad(model, img_tensor, target_class, num_samples=50, noise_level=0.2):
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
        class_score = output[0, target_class]
        class_score.backward()

        if noisy_img.grad is not None:
            smooth_grad += noisy_img.grad.abs()

    smooth_grad /= num_samples
    saliency = smooth_grad.squeeze().cpu().numpy()
    saliency = np.max(saliency, axis=0)
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
    saliency = 1.0 - saliency
    return saliency

image_dir = 'imageNet_test_images'
#image_dir = 'consistency_test'
output_dir = "smoothgrad_results"
#output_dir = "smoothgrad_consistency"
os.makedirs(output_dir, exist_ok=True)

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
        saliency_map = generate_smoothgrad(model, img_tensor, class_idx)

        axes[i + 1].imshow(saliency_map, cmap="gray")
        axes[i + 1].axis("off")
        label = top5_classes[i]
        prob = top5_probs[i]
        axes[i + 1].set_title(f"{label}\n{prob:.2f}")

    end_time = time.time()
    duration = end_time - start_time
    print(f"Explanation time for {img_name}: {duration:.2f} seconds")

    save_path = os.path.join(output_dir, f"{os.path.splitext(img_name)[0]}.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
