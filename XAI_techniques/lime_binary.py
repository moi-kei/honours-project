import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from lime import lime_image
from torchvision import models, transforms
from PIL import Image
from skimage.segmentation import mark_boundaries

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

def batch_predict(images):
    model.eval()
    batch = torch.stack([transforms.ToTensor()(Image.fromarray(img)).float() for img in images]).to(device)
    batch = (batch - torch.tensor(mean[:, None, None], dtype=torch.float32, device=device)) / torch.tensor(std[:, None, None], dtype=torch.float32, device=device)
    with torch.no_grad():
        outputs = model(batch)
    probs = torch.sigmoid(outputs).cpu().numpy()
    probs = np.concatenate([1 - probs, probs], axis=1)
    return probs

image_dir = 'test_images/brain_tumour_test_images'
#output_dir = "results/brain_tumour/pretrained/lime"
output_dir = "results/brain_tumour/untrained/lime"
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
    confidence = prob if pred_class == 1 else 1 - prob
    label = class_names[pred_class]
    print(f'Predicted for {img_name}: {label} ({confidence:.2f})')

    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(raw_image, batch_predict, top_labels=2, hide_color=0, num_samples=5000)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(raw_image)
    axes[0].axis("off")
    axes[0].set_title("Original Image")

    temp, mask = explanation.get_image_and_mask(pred_class, positive_only=False, num_features=5, hide_rest=False)
    lime_image_result = mark_boundaries(temp, mask)
    axes[1].imshow(lime_image_result)
    axes[1].axis("off")
    axes[1].set_title(f"{label} ({confidence:.2f})")

    save_path = os.path.join(output_dir, f"{os.path.splitext(img_name)[0]}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"LIME visualization saved to: {save_path}")
