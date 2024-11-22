import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image

def load_and_preprocess_image(img_path, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

def extract_features(img_path):
    model = ResNet50(weights='imagenet', include_top=False)
    img_preprocessed = load_and_preprocess_image(img_path)
    features = model.predict(img_preprocessed)
    return features

def overlay_feature_map(original_img, feature_map, alpha=0.6, cmap='jet'):
    feature_map_normalized = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min() + 1e-8)
    heatmap = cv2.resize(feature_map_normalized, (original_img.shape[1], original_img.shape[0]))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(original_img, 1 - alpha, heatmap_colored, alpha, 0)
    return overlay

def visualize_multiple_overlays(img_path, features, num_feature_maps=4):
    img = image.load_img(img_path)
    img_array = image.img_to_array(img)
    img_array = cv2.cvtColor(img_array.astype('uint8'), cv2.COLOR_BGR2RGB)
    num_channels = features.shape[-1]
    num_to_plot = min(num_feature_maps, num_channels)
    cols = 2
    rows = (num_to_plot + 1) // cols
    plt.figure(figsize=(10, rows * 5))
    for i in range(num_to_plot):
        feature_map = features[0, :, :, i]
        overlay_img = overlay_feature_map(img_array, feature_map)
        plt.subplot(rows, cols, i + 1)
        plt.title(f"Feature Map {i + 1}")
        plt.imshow(overlay_img)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    img_path = 'stingray.jpg'
    features = extract_features(img_path)
    visualize_multiple_overlays(img_path, features, num_feature_maps=8)
