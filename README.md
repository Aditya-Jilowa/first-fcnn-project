Handwritten Digit Recognition with Neural Network
Overview
This project implements a two-layer feedforward neural network for handwritten digit recognition using the MNIST dataset (or a similar dataset of 28x28 grayscale images). The neural network is built from scratch in Python using NumPy for computations and Matplotlib for visualization. The project includes training a model, testing its predictions, and saving/loading model parameters for reuse.
Project Structure
The project consists of two main Python scripts:

model.py: Implements the neural network, including data loading, training, and evaluation.
test.py: Loads trained model parameters and tests predictions on random test images.

Dataset
The model uses the following dataset files (not included in the code):

train_X.csv: Training images (784 features per image, representing 28x28 pixels).
train_label.csv: One-hot encoded labels for training data (10 classes, digits 0–9).
test_X.csv: Test images (same format as training data).
test_label.csv: One-hot encoded labels for test data.

Data Format:

Images are loaded as a matrix where each column is a flattened 28x28 image (784xN for N images).
Labels are one-hot encoded (10xN, where each column has a 1 for the correct digit and 0s elsewhere).

Neural Network Architecture

Input Layer: 784 nodes (flattened 28x28 image pixels).
Hidden Layer: 1000 nodes with ReLU activation.
Output Layer: 10 nodes (one per digit, 0–9) with softmax activation.
Loss Function: Cross-entropy loss.
Optimizer: Gradient descent with a learning rate of 0.003.
Training: 100 iterations (epochs) over the training data.

Implementation Details
model.py
This script handles data loading, model training, and parameter saving:

Data Loading: Loads training and test data using np.loadtxt. Displays a random training image for verification.
Activation Functions:
tanh(x): Hyperbolic tangent (not used in the final model but included).
relu(x): Rectified Linear Unit for the hidden layer.
softmax(x): Normalizes output probabilities for classification.


Derivatives: derivative_relu and derivative_tanh for backpropagation.
Parameter Initialization: Weights (w1, w2) are initialized with small random values (randn * 0.01); biases (b1, b2) are initialized to zeros.
Forward Propagation: Computes hidden and output layer activations.
Cost Function: Calculates cross-entropy loss between predicted and actual labels.
Backward Propagation: Computes gradients for weights and biases using the chain rule.
Parameter Update: Updates weights and biases using gradient descent.
Training Loop:
Runs for 100 iterations with a hidden layer size of 1000 and learning rate of 0.003.
Prints cost every 10 iterations.
Plots the cost over iterations using Matplotlib.


Evaluation: Tests the model on a random test image and prints the predicted digit.
Parameter Saving: Saves weights and biases to model_parameters.npz.

test.py
This script loads the trained model and tests it on random test images:

Loads parameters from model_parameters.npz.
Defines activation functions (tanh, relu, softmax) and forward_propagation (same as model.py).
Randomly selects 10 test images, displays each with Matplotlib, and shows the predicted digit as the title.

Requirements

Python 3.x
NumPy
Matplotlib

Install dependencies using:
pip install numpy matplotlib

Usage

Prepare the Dataset:

Ensure train_X.csv, train_label.csv, test_X.csv, and test_label.csv are in the project directory.
The dataset should contain flattened 28x28 images and one-hot encoded labels.


Train the Model:

Run model.py to train the neural network:python model.py


The script will:
Load and display a random training image.
Train the model for 100 iterations, printing the cost every 10 iterations.
Plot the training cost over iterations.
Test a random test image and print the predicted digit.
Save parameters to model_parameters.npz.




Test the Model:

Run test.py to load the trained model and test predictions:python test.py


The script will display 10 random test images with their predicted digits.



Output

Training:
Prints the shapes of training and test data.
Displays a random training image.
Shows cost values during training.
Plots the cost vs. iteration graph.
Tests a random test image and prints the predicted digit.
Saves model parameters.


Testing:
Loads saved parameters.
Displays 10 random test images with predicted digit labels.



Example Output
shape of X_train : (784, 60000)
shape of Y_train : (10, 60000)
shape of X_test : (784, 10000)
shape of Y_test : (10, 10000)
cost after 0 iteration is : 2.302585
cost after 10 iteration is : 0.892134
...
out model says, it is : 7
Model parameters saved to model_parameters.npz

Limitations

The model uses a simple two-layer architecture, which may not achieve state-of-the-art accuracy on MNIST.
Training is limited to 100 iterations, which may not be sufficient for convergence.
No validation set or accuracy metrics are computed.
The heuristic evaluation in test.py is basic, displaying only 10 random predictions without overall accuracy.

Potential Enhancements

Add a validation set to monitor overfitting.
Compute and display training and test accuracy.
Implement batch processing for faster training.
Experiment with different architectures (e.g., more layers, different activation functions).
Add regularization (e.g., L2) to prevent overfitting.
Include a confusion matrix or other metrics for evaluation.

Notes

Ensure the dataset files are correctly formatted and present in the working directory.
The model assumes the input data is preprocessed (normalized pixel values, one-hot encoded labels).
The project is designed for educational purposes, demonstrating a basic neural network implementation.
