

# Digit Classifier using Convolutional Neural Network (CNN)

This project implements a digit classification system using a neural network trained on the MNIST dataset. The model classifies handwritten digits in real-time using a webcam. The system captures frames, processes them to detect and classify digits, and displays the predictions on the screen.

## Project Overview

The system is built using the following components:
- **MNIST Dataset**: A collection of 28x28 pixel images of handwritten digits.
- **TensorFlow & Keras**: For building and training the neural network.
- **OpenCV**: For handling video input from the camera and processing the images.
- **Matplotlib and Seaborn**: For visualizing data and model performance.

## Requirements

To run this project, you need to install the following dependencies:

- **TensorFlow**: For building and training the deep learning model.
- **Keras**: High-level API for building neural networks.
- **OpenCV**: For capturing webcam feed and processing frames.
- **Numpy**: For numerical operations.
- **Matplotlib & Seaborn**: For data visualization.

You can install these dependencies using the following commands:

pip install tensorflow opencv-python numpy matplotlib seaborn


## How It Works

### Step 1: Preprocess the Data
- The MNIST dataset is loaded and normalized so that pixel values are between 0 and 1.
- The model is trained on the dataset for 5 epochs with a batch size of 32.

### Step 2: Build and Train the Model
- The model is a simple feedforward neural network with 3 layers:
  - Flatten layer to reshape the input (28x28) into a 1D vector.
  - Two fully connected layers with ReLU activation.
  - Output layer with 10 neurons (for each digit) using softmax activation.

### Step 3: Model Evaluation
- After training, the model is evaluated on the test dataset.
- The accuracy and loss are printed.

### Step 4: Real-Time Digit Recognition
- OpenCV is used to capture video frames from the webcam.
- A region of interest (ROI) is defined in the center of the frame where a digit is likely to appear.
- The captured region is preprocessed, resized to 28x28 pixels, and normalized before being passed to the model for prediction.
- The predicted digit is displayed on the frame, along with a green rectangle around the detected area.

### Step 5: Save and Load Model
- The trained model is saved to a file (`digit_classifier.h5`) for later use.
- The model can be loaded for real-time predictions.

### Step 6: Run the Application
- The webcam feed is displayed in a window. The digit prediction is continuously updated as the camera captures frames.
- Press 'q' to exit the application.

## How to Use

1. Clone this repository or download the project files.
2. Install the required dependencies as described above.
3. Run the main script to start the webcam and begin digit classification:
    ```bash
    python main.py
    ```
4. The system will show a live feed from your webcam. Write a digit on a piece of paper and hold it in front of the camera to see the predicted digit in real time.
5. To exit the application, press the 'q' key.

## Model Performance

The model achieves a test accuracy of approximately **98%** after training for 5 epochs on the MNIST dataset.

## Project by

**Rohan Chatse**  
*Data Scientist*  
Email: [rohancrchatse@gmail.com](mailto:rohan@gmail.com)
`

