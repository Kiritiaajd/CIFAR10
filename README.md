# CIFAR-10 Image Classification Project

This project involves classifying images from the **CIFAR-10** dataset, which is a well-known dataset used for training machine learning models on image classification tasks. The dataset contains 60,000 32x32 color images in 10 different classes, with each class having 6,000 images. The classes include animals, vehicles, and other everyday objects.

## Project Overview

In this project, I implemented a **Convolutional Neural Network (CNN)** using **PyTorch** to classify the images in the CIFAR-10 dataset into their respective categories. The goal was to preprocess the data, build a CNN model, train the model on the CIFAR-10 dataset, and evaluate its performance.

## Dataset

The **CIFAR-10** dataset is split into two parts:
- **Training Set**: 50,000 labeled images
- **Test Set**: 10,000 labeled images

Each image is 32x32 pixels in size and is categorized into one of the following 10 classes:

1. Airplane
2. Automobile
3. Bird
3. Cat
4. Deer
5. Dog
6. Frog
7. Horse
8. Ship
9. Truck

## Project Structure

The project is organized into the following main sections:

### 1. Data Preprocessing:
- Loading the CIFAR-10 dataset using `torchvision.datasets`.
- Normalizing the dataset to standardize pixel values for model training.

### 2. Model Building:
- Building a **Convolutional Neural Network (CNN)** with several convolutional layers, pooling layers, and fully connected layers.
- Using **ReLU** activations and **softmax** for multi-class classification.

### 3. Training:
- Training the model using the **Adam Optimizer** and **Cross-Entropy Loss**.
- Monitoring the model's accuracy during training using the **training loss** and **validation accuracy**.

### 4. Evaluation:
- Evaluating the trained model on the test dataset.
- Displaying accuracy metrics such as overall accuracy, precision, recall, and F1-score.
- Visualizing the confusion matrix to understand model performance across the classes.

### 5. Model Saving and Loading:
- Saving the trained model for future use.
- Loading the saved model and making predictions on new images.

## Requirements

To run this project, you'll need the following Python libraries:

- `torch` (PyTorch)
- `torchvision`
- `matplotlib`
- `numpy`
- `PIL`



Results
After training the model, the CNN achieved an accuracy of around 90% on the CIFAR-10 test set. The model's performance was assessed using a confusion matrix and classification report, which provided insights into how well the model performed across different classes.

Challenges Faced
Overfitting: During the early stages of training, the model showed signs of overfitting. This was addressed by using data augmentation, dropout, and early stopping techniques.
Computational Resources: Training deep neural networks on the CIFAR-10 dataset requires substantial computational power, especially for larger models. Using GPUs helped significantly reduce training time.
Conclusion
This project demonstrates the application of a Convolutional Neural Network (CNN) for image classification using the CIFAR-10 dataset. By following best practices in model building and evaluation, the project achieved good performance in classifying images into one of 10 categories.

Future Work
Model Improvements: Experiment with deeper architectures like ResNet or VGG to improve accuracy.
Transfer Learning: Use pre-trained models and fine-tune them on the CIFAR-10 dataset to achieve better results.
Data Augmentation: Increase the size and diversity of the training set through additional data augmentation techniques.
You can install them via `pip`:

```bash
pip install torch torchvision matplotlib numpy pillow
