# Tamil Handwritten Character Recognition

A deep learning–based system to recognize handwritten Tamil characters using Convolutional Neural Networks (CNNs), with support for real-world handwritten inputs and character segmentation.

---

## Project Description

Tamil handwritten character recognition is challenging due to complex character structures, variations in handwriting styles, and noise in scanned or photographed documents.  
This project implements an end-to-end recognition system trained on a large Tamil handwritten dataset and extended with OpenCV-based segmentation to recognize characters from handwritten words or lines.

The model classifies 156 unique Tamil characters and performs well on both standard and self-collected handwritten samples.

---

## Features

- CNN-based handwritten Tamil character recognition  
- Supports real handwritten inputs  
- Character segmentation using OpenCV  
- High accuracy with detailed evaluation metrics  
- Works on scanned images and photographs  

---

## Methodology

### Dataset Preparation
- Class-wise dataset organization  
- Train/Test split: 80/20  

### Preprocessing
- Grayscale conversion  
- Resize to 32×32  
- Normalization  
- Label encoding  

### Model Architecture
- Convolution and pooling layers  
- Dropout for regularization  
- Dense layers with Softmax output  

### Training
- Optimizer: Adam  
- Loss Function: Categorical Cross-Entropy  

### Segmentation and Prediction
- Adaptive thresholding  
- Contour detection  
- Bounding box extraction  

---

## Dataset

- Standard Tamil Handwritten Dataset  
- Self-Collected Handwritten Samples  
- Total Classes: 156 Tamil characters  

---

## Results

- High accuracy on the test dataset  
- Strong generalization on real handwritten inputs  

### Evaluation Metrics
- Accuracy  
- Precision  
- Recall  
- F1-Score  
- Confusion Matrix  

---

## Tech Stack

- Python  
- TensorFlow / Keras  
- OpenCV  
- NumPy  
- Matplotlib  
- Google Colab  

---

## Code

Google Colab Notebook:  
https://colab.research.google.com/drive/1aSv93fgjRxhVF7esMxojtnl8CkE3ejL2

---

## Future Enhancements

- Word and sentence-level recognition  
- Support for compound Tamil characters  
- OCR system integration  
- Mobile and web deployment  
- Transfer learning with advanced architectures 
