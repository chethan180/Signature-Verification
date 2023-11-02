# Signature Forgery Detection using Siamese Neural Networks

This project trains Siamese neural networks to detect forged signatures using two different base network architectures.

## Dataset
The [CEDAR handwritten signature dataset](https://www.kaggle.com/datasets/ishanikathuria/handwritten-signature-datasets) from Kaggle is used. It contains 55 folders with 48 images in each - 24 original signatures and 24 forged signatures.

## Preprocessing
The images are loaded, resized to 224x224 and normalized before feeding to the model.

## Model 1 (Custom CNN base)
A Siamese neural network architecture with a custom Convolutional Neural Network base is used. The base CNN contains:
- Conv2D layers
- MaxPooling2D layers  
- Flatten layer
- Dense layer

The embeddings from the base CNN are compared using L1 distance and passed through a Dense sigmoid layer for classification.  

Binary crossentropy loss and adam optimizer are used.

Early stopping is implemented to stop training at 95% validation accuracy.

## Model 2 (VGG19 base)
A Siamese neural network architecture with a VGG19 base is used. The first 5 layers of VGG19 are frozen. 

The flattened embeddings from VGG19 are compared using L1 distance and passed through a Dense sigmoid layer.

Rest of the model is same as Model 1.

## Training
The image pairs are converted to NumPy arrays and labels are assigned 0 for original-original pairs and 1 for original-forged pairs.

The data is split 70-30 for training and validation. 

Model 1 is trained for 10 epochs with batch size 16. 

Model 2 is trained for 10 epochs with batch size 8.

## Evaluation
The trained model is evaluated on the test set. The predictions indicate the score between 0 and 1. Values closer to 0 indicates they are similar while values closer to 1 indicate they are dissimilar signature pair.

# Accuracy: 0.98
# Precision: 0.9615384615384616
# Recall: 1.0
# F1-Score: 0.9803921568627451

## Usage
The models can be loaded and used to predict similarity between new signature pairs:

```
model = load_model('model_name')

img1 = np.array([preprocess_image(image1_path)]) 
img2 = np.array([preprocess_image(image2_path)])

prediction = model.predict([img1, img2]) 
print(prediction) # Predicted similarity score
```

## Future Work
- Experiment with different CNN architectures like ResNet50 for the base network.
- Use more data augmentation to handle class imbalance better. 
- Try different distance metrics like cosine distance.

