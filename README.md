# Cats vs. Dogs Classification

This repository demonstrates a deep learning solution for classifying images of cats and dogs. The main work is documented in the Jupyter Notebook, which covers the complete workflow—from data preprocessing to model training and evaluation. Pre-trained models (the trained weights) are available in the Releases section.

---

## Project Overview

- **Jupyter Notebook (`cat_dog_classification.ipynb`):**  
  This notebook contains all the code and steps required for the cat vs. dog classification project. It is organized into clear sections that guide you through the process of:
  
  - **Data Preprocessing:**  
    Loading and preparing image data, including augmentation using tools like `ImageDataGenerator`.
    
  - **Model Building:**  
    Constructing deep learning models with transfer learning. The notebook demonstrates the use of architectures like Xception and VGG16 (the specific details of each model are built into the code).
    
  - **Training & Evaluation:**  
    Setting up training parameters (e.g., optimizer, loss function, metrics) and running experiments. The notebook includes code for training the models and evaluating their performance.
    
  - **Prediction:**  
    A step-by-step guide to loading the trained model and making predictions on new images. This section shows you how to preprocess an input image and interpret the model’s output.
    
- **Trained Models:**  
  The trained model files are attached in the Releases section. You can download these files and load them directly in your projects to make predictions without having to retrain the models.

---

## Getting Started

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/harshhmaniya/cats-dogs-classification-97-accuracy.git
   ```

2. **Open the Notebook:**  
   Launch `cat_dog_classification.ipynb` using Jupyter Notebook or any compatible IDE to explore and run the code.

3. **Download Trained Models:**  
   Visit the [Releases](https://github.com/harshhmaniya/cats-dogs-classification-97-accuracy/releases) page to download the trained model files attached to each release.

---

## How to Use

### Loading a Saved Model

Each model has been saved using the Keras native format (a `.keras` file). To load a model in your own project, use:

```python
from tensorflow.keras.models import load_model

# For example, to load the Xception model:
model = load_model("Xception_97_acc.keras")
```

### Making Predictions

Once the model is loaded, use it to predict on new images as follows:

```python
import numpy as np
from tensorflow.keras.preprocessing import image

# Define image dimensions (should match IMG_HEIGHT and IMG_WIDTH used during training)
IMG_HEIGHT, IMG_WIDTH = 224, 224

# Load and preprocess your image
img = image.load_img("your_image.jpg", target_size=(IMG_HEIGHT, IMG_WIDTH))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Get prediction (output > 0.5 implies one class, otherwise the other)
prediction = model.predict(img_array)
print("Dog" if prediction > 0.5 else "Cat")
```
---

## License

This project is licensed under the MIT License. See the [LICENSE](https://github.com/harshhmaniya/cats-dogs-classification-97-accuracy/blob/main/LICENSE) file for details.

---


## Contact

For issues, improvements, or suggestions, please open an issue in this repository.

### Author
- **Harsh Maniya**  
- [LinkedIn](https://linkedin.com/in/harsh-maniya)
- [GitHub](https://github.com/harshhmaniya)


