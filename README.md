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

## Example: Sticker Overlay
```python
>>> img_proc = PremiumImageProcessing()
>>> img_sticker = img_read_helper('img/square_6x6.png')
>>> img_back = img_read_helper('img/test_image_32x32.png')
>>> x, y = (3, 3)
>>> img_exp = img_read_helper('img/exp/test_image_32x32_sticker.png')
>>> img_combined = img_proc.sticker(img_sticker, img_back, x, y)
>>> img_combined.pixels == img_exp.pixels  # Check sticker output
True
>>> img_save_helper('img/out/test_image_32x32_sticker.png', img_combined)
