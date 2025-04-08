import json
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import shap
from torchvision import models, transforms
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).to(device)
model.eval()

url = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
with open(shap.datasets.cache(url)) as file:
    class_names = [v[1] for v in json.load(file).values()]

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
        outputs = model(x_tensor)
    return outputs.cpu().numpy()

image_dir = 'imageNet_test_images'
os.makedirs("shap_results", exist_ok=True)
#image_dir = 'consistency_test'
#os.makedirs("shap_consistency", exist_ok=True)

for img_name in os.listdir(image_dir):
    img_path = os.path.join(image_dir, img_name)
    if not os.path.isfile(img_path):
        continue
    
    original_image, img_tensor = load_image(img_path)
    
    with torch.no_grad():
        preds = model(img_tensor).cpu().numpy()
    
    print(f'Predicted for {img_name}:', class_names[np.argmax(preds)])
    X = img_tensor.cpu().numpy().transpose(0, 2, 3, 1)
    masker_blur = shap.maskers.Image("blur(128,128)", X[0].shape)
    explainer_blur = shap.Explainer(f, masker_blur, output_names=class_names)
    shap_values_fine = explainer_blur(X, max_evals=5000, batch_size=50, outputs=shap.Explanation.argsort.flip[:5])
    plt.figure(figsize=(10, 5))
    shap.image_plot(shap_values_fine, show=False)
    save_path = os.path.join("shap_results", f"{os.path.splitext(img_name)[0]}.png")
    #save_path = os.path.join("shap_consistency", f"{os.path.splitext(img_name)[0]}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()