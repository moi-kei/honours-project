import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import shap
from torchvision import models, transforms
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_custom_model(model_path):
    model = models.resnet50()
    model.fc = torch.nn.Sequential(
        torch.nn.Linear(2048, 256),
        torch.nn.ReLU(),
        torch.nn.Dropout(),
        torch.nn.Linear(256, 1)
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
    return img, img_tensor

def f(x):
    x_tensor = torch.tensor(x, dtype=torch.float32, device=device).permute(0, 3, 1, 2)
    with torch.no_grad():
        logits = model(x_tensor)
        probs = torch.sigmoid(logits)
        probs = torch.cat([1 - probs, probs], dim=1)
    return probs.cpu().numpy()

image_dir = 'test_images/brain_tumour_test_images'
#output_dir = "results/brain_tumour/pretrained/shap"
output_dir = "results/brain_tumour/untrained/shap"
os.makedirs(output_dir, exist_ok=True)

for img_name in os.listdir(image_dir):
    img_path = os.path.join(image_dir, img_name)
    if not os.path.isfile(img_path):
        continue

    original_image, img_tensor = load_image(img_path)

    with torch.no_grad():
        logits = model(img_tensor).cpu().numpy()
    pred_class = 1 if logits[0][0] > 0 else 0
    print(f'Predicted for {img_name}:', class_names[pred_class])

    X = img_tensor.cpu().numpy().transpose(0, 2, 3, 1)
    masker_blur = shap.maskers.Image("blur(128,128)", X[0].shape)
    explainer_blur = shap.Explainer(f, masker_blur, output_names=class_names)
    shap_values_fine = explainer_blur(X, max_evals=5000, batch_size=50, outputs=shap.Explanation.argsort.flip[:1])

    plt.figure(figsize=(10, 5))
    shap.image_plot(shap_values_fine, show=False)
    save_path = os.path.join(output_dir, f"{os.path.splitext(img_name)[0]}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()