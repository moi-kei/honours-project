import torch
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import os
import json
from PIL import Image
from torchvision.models import ResNet50_Weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).to(device)
model.eval()

target_layers = {
    "layer2": model.layer2,
    "layer3": model.layer3,
    "layer4": model.layer4
}

activations = {}

def register_hooks():
    handles = []
    for layer_name, layer in target_layers.items():
        for i, block in enumerate(layer):
            layer_id = f"{layer_name}[{i}]"
            def hook_fn(module, input, output, name=layer_id):
                activations[name] = output.detach()
            hook = block.register_forward_hook(hook_fn)
            handles.append(hook)
    return handles

transform = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0).to(device)

def save_feature_maps(activations, output_root, resize_to=(256, 256), activation_threshold=1e-3):
    os.makedirs(output_root, exist_ok=True)
    
    for layer_id, feature_maps in activations.items():
        feature_maps = feature_maps.squeeze(0).cpu() 
        num_filters = feature_maps.shape[0]

        layer_dir = os.path.join(output_root, *layer_id.replace("[", "/").replace("]", "").split("/"))
        os.makedirs(layer_dir, exist_ok=True)

        for i in range(num_filters):
            feature_map = feature_maps[i].numpy()
            mean_activation = np.mean(feature_map)
            if np.abs(mean_activation) < activation_threshold:
                continue 
            epsilon = 1e-8
            feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min() + epsilon)
            feature_map = np.nan_to_num(feature_map, nan=0.0, posinf=255, neginf=0)
            feature_map = (feature_map * 255).astype(np.uint8)
            img = Image.fromarray(feature_map)
            img = img.resize(resize_to) 
            img.save(os.path.join(layer_dir, f"filter_{i}.png"))
    
    print(f"Saved feature maps in {output_root}")

def classify_image(input_tensor):
    with torch.no_grad():
        output = model(input_tensor)
        softmax = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted_class_idx = torch.max(softmax, 1)
        with open('imagenet_class_index.json') as f:
            class_idx = json.load(f)
        predicted_class = class_idx[str(predicted_class_idx.item())][1]      
        return predicted_class, confidence.item()

def process_image(image_path):
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    output_root = os.path.join("feature_maps", image_name)
    hooks = register_hooks()
    input_tensor = load_image(image_path)
    with torch.no_grad():
        model(input_tensor)
    save_feature_maps(activations, output_root)
    predicted_class, confidence = classify_image(input_tensor)
    print(f"Predicted: {predicted_class} with confidence: {confidence * 100:.2f}%")
    for hook in hooks:
        hook.remove()

image_path = "cheeseburger.jpg"
process_image(image_path)
