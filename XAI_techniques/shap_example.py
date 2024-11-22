import json
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import shap

model = ResNet50(weights="imagenet")

url = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
with open(shap.datasets.cache(url)) as file:
    class_names = [v[1] for v in json.load(file).values()]

def load_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def f(x):
    tmp = x.copy()
    preprocess_input(tmp)
    return model(tmp)

img_path = 'tiger-shark.jpeg'
img_array = load_image(img_path)

preds = model.predict(img_array)
print('Predicted:', decode_predictions(preds, top=3)[0])

masker_blur = shap.maskers.Image("blur(128,128)", img_array[0].shape)

explainer_blur = shap.Explainer(f, masker_blur, output_names=class_names)

shap_values_fine = explainer_blur(img_array, max_evals=5000, batch_size=50, outputs=shap.Explanation.argsort.flip[:4])

shap.image_plot(shap_values_fine)

plt.savefig('shap_visualization.png', dpi=300, bbox_inches='tight')
plt.show()


