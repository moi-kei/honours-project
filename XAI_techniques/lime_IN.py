import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from lime import lime_image
from torchvision import models, transforms
from PIL import Image
from skimage.segmentation import mark_boundaries
import json
import urllib.request

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

def batch_predict(images):
    model.eval()
    batch = torch.stack([transforms.ToTensor()(Image.fromarray(img)).float() for img in images]).to(device)
    batch = (batch - torch.tensor(mean[:, None, None], dtype=torch.float32, device=device)) / torch.tensor(std[:, None, None], dtype=torch.float32, device=device)
    with torch.no_grad():
        preds = model(batch)
    return preds.cpu().numpy()

image_dir = 'imageNet_test_images'
os.makedirs("lime_results", exist_ok=True)
#image_dir = 'consistency_test'
#os.makedirs("lime_consistency", exist_ok=True)

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

    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(raw_image, batch_predict, top_labels=5, hide_color=0, num_samples=5000)

    fig, axes = plt.subplots(1, 6, figsize=(18, 5))
    axes[0].imshow(raw_image)
    axes[0].axis("off")
    axes[0].set_title("Original")

    for i, class_idx in enumerate(explanation.top_labels[:5]):
        temp, mask = explanation.get_image_and_mask(class_idx, positive_only=False, num_features=5, hide_rest=False)
        lime_image_result = mark_boundaries(temp, mask)
        class_label = class_names[class_idx]
        class_prob = probs[class_idx]
        axes[i + 1].imshow(lime_image_result)
        axes[i + 1].axis("off")
        axes[i + 1].set_title(f"{class_label}\n{class_prob:.2f}")

    save_path = os.path.join("lime_results", f"{os.path.splitext(img_name)[0]}.png")
    #save_path = os.path.join("lime_consistency", f"{os.path.splitext(img_name)[0]}.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()