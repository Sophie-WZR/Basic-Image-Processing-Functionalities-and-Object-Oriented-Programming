# Basic Image Processing with OOP

This project focuses on building basic image processing tools and a K-Nearest Neighbors (KNN) classifier, using object-oriented programming (OOP) principles. The project allows users to apply several image transformations, perform image classification with KNN, and visualize images with interactive features like zoom and hover using a GUI built in Python.

## Features

1. **Image Processing:**
   - Apply basic image transformations such as:
     - **Color negation**
     - **Grayscale**
     - **Rotation by 180 degrees**
     - **Brightness adjustment**
     - **Blur effect**
     - **Edge highlighting**
     - **Chroma key (green screen) effect**
   - Implemented using OOP principles.

2. **KNN Classifier:**
   - A K-Nearest Neighbors classifier that uses pixel data for image classification.
   - Supports training on labeled image datasets and predicting labels of new images.
   - Calculates the Euclidean distance between images and identifies the label with the most votes from the K-nearest neighbors.

3. **Interactive Image Viewer:**
   - View and zoom in/out on images using the GUI.
   - Hover to display the RGB color values of pixels.
   - Scroll and pan across images.
   - Built using the Tkinter library and PIL (Python Imaging Library).

## Installation

To run the project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Basic-Image-Processing-with-OOP.git
   cd Basic-Image-Processing-with-OOP
