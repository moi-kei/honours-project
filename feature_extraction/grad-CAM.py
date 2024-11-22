import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model

def load_and_preprocess_image(img_path, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array), image.img_to_array(image.load_img(img_path))

def extract_features_and_model(img_path):
    base_model = ResNet50(weights='imagenet')
    model = Model(inputs=base_model.input, outputs=base_model.get_layer("conv5_block3_out").output)
    img_preprocessed, original_img = load_and_preprocess_image(img_path)
    features = model.predict(img_preprocessed)
    return features, original_img

def highlight_feature_map(original_img, feature_map):
    feature_map_resized = cv2.resize(feature_map, (original_img.shape[1], original_img.shape[0]))
    mask = (feature_map_resized > np.mean(feature_map_resized)).astype(np.uint8)
    highlighted = cv2.bitwise_and(original_img, original_img, mask=mask)
    return highlighted

def visualize_highlighted_features(img_path, features, original_img, num_feature_maps=4):
    num_channels = features.shape[-1]
    num_to_plot = min(num_feature_maps, num_channels)
    cols = 4
    rows = (num_to_plot + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 3))
    axes = axes.flatten()
    for i in range(num_to_plot):
        feature_map = features[0, :, :, i]
        highlighted_img = highlight_feature_map(original_img, feature_map)
        axes[i].imshow(highlighted_img.astype(np.uint8))
        axes[i].set_title(f"Feature Map {i + 1}")
        axes[i].axis('off')
    for i in range(num_to_plot, len(axes)):
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    img_path = 'stingray.jpg'
    features, original_img = extract_features_and_model(img_path)
    visualize_highlighted_features(img_path, features, original_img, num_feature_maps=60)
