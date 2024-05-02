# Flower Classification using Convolutional Neural Networks (CNN)

This repository contains code for building a flower classification model using Convolutional Neural Networks (CNN) with TensorFlow and Keras. The model classifies images of flowers into five categories: roses, daisies, dandelions, sunflowers, and tulips.

## Dataset
The dataset used in this project is the [flower_photos](https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz) dataset, which contains images of flowers belonging to different categories.

## Prerequisites
- Python 3.x
- TensorFlow
- NumPy
- OpenCV (cv2)
- PIL (Python Imaging Library)
- Matplotlib
- scikit-learn
- seaborn

You can install the required dependencies using pip:

```bash
pip install tensorflow opencv-python-headless pillow matplotlib scikit-learn seaborn
```

## Usage
1. Clone this repository:

```bash
git clone https://github.com/akshaykapoor0209/flower-classification-cnn.git
```

2. Navigate to the project directory:

```bash
cd flower-classification-cnn
```

3. Download the dataset:

```bash
python flowersClassification.py
```

4. Train the model:

```bash
python flowersClassification.py
```

5. Evaluate the model:

```bash
python flowersClassification.py
```

## Results
The trained model achieves good performance on the training data. However, there might be overfitting issues as the performance on the test data could be lower. To address overfitting, data augmentation techniques such as random zoom, rotation, and translation are implemented.
