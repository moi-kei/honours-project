import json
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import shap

# Load pre-trained model
model = ResNet50(weights="imagenet")

# Get ImageNet 1000 class names
url = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
with open(shap.datasets.cache(url)) as file:
    class_names = [v[1] for v in json.load(file).values()]

# Function to load and preprocess the image
def load_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# Function to get model output
def f(x):
    tmp = x.copy()
    preprocess_input(tmp)
    return model(tmp)

# Input your own image path here
img_path = 'tiger-shark.jpeg'  # Replace with your own image path
img_array = load_image(img_path)

# Make a prediction
preds = model.predict(img_array)
print('Predicted:', decode_predictions(preds, top=3)[0])

# Define a masker that uses blur for masking
masker_blur = shap.maskers.Image("blur(128,128)", img_array[0].shape)

# Create an explainer with the model and blur masker
explainer_blur = shap.Explainer(f, masker_blur, output_names=class_names)

# Explain the image using 5000 evaluations
shap_values_fine = explainer_blur(img_array, max_evals=5000, batch_size=50, outputs=shap.Explanation.argsort.flip[:4])

# Visualize the SHAP values
shap.image_plot(shap_values_fine)

# Save the plot to a file
plt.savefig('shap_visualization.png', dpi=300, bbox_inches='tight')
plt.show()







