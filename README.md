Here is a detailed README.md for the Python notebook provided:

# Signature Forgery Detection using Siamese Neural Networks

This project trains a Siamese neural network to detect forged signatures.

## Dataset
The [CEDAR handwritten signature dataset](https://www.kaggle.com/datasets/ishanikathuria/handwritten-signature-datasets) from Kaggle is used. It contains 55 folders with 24 images in each - 12 original signatures and 12 forged signatures.

## Preprocessing
The images are loaded, resized to 224x224 and normalized before feeding to the model.

## Model
A Siamese neural network architecture is used. It contains a base Convolutional Neural Network which generates embeddings for the input images. These embeddings are compared using a distance metric and fed to a Dense layer for classification.

The base CNN contains:
- Conv2D layers
- MaxPooling2D layers
- Flatten layer
- Dense layer 

The distance metric used is L1 distance between the embeddings.

The model is compiled with binary crossentropy loss and adam optimizer.

## Training
The image pairs are converted to numpy arrays and labels are assigned 0 for original-original pairs and 1 for original-forged pairs.

The data is split into 70% training and 30% testing.

The model is trained for 10 epochs with a batch size of 16. Early stopping is implemented to stop training when validation accuracy reaches 95%.

## Evaluation
The trained model is evaluated on the test set. The predictions indicate the similarity score between 0 and 1. Values closer to 0 indicate a forged signature while values closer to 1 indicate an original signature pair.

## Usage
The model can be loaded and used to predict similarity between new signature pairs:

```
model = load_model('final') 

img1 = np.array([preprocess_image(image1_path)])
image2 = np.array([preprocess_image(image2_path) ])

prediction = model.predict([img1, img2])
print(prediction) # Predicted similarity score
```

A lower score indicates the signatures are likely forged while a higher score indicates original signatures.

## Future Work
- Experiment with different CNN architectures for the base network.
- Try different distance metrics like euclidean distance.
- Use more data augmentation to handle class imbalance better.

Let me know if you need any clarification or have additional questions!
