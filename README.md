# Coffee Roasting Classification using Neural Network
This is a small machine learning project I built while learning the basics of **Neural Networks and TensorFlow**.
The idea is simple: based on the **temperature** and **roasting time**, the model predicts whether the coffee is likely to be a **Good Roast** or a **Bad Roast**.

## What I learned
While building this project, I practiced:
* Creating and working with a dataset using NumPy
* Visualizing data using Matplotlib
* Normalizing input features
* Creating a neural network using TensorFlow/Keras
* Compiling and training a model
* Understanding weights and biases
* Making predictions on new data
* Converting prediction probabilities into 0/1 classes
* Visualizing the model's predictions

## Neural Network
The model has:
2 Input Features
      ↓
3 Neurons (Hidden Layer)
      ↓
1 Output Neuron

The two input features are:

* Temperature (°C)
* Roasting Duration (minutes)

The output represents:
1 → Good Roast
0 → Bad Roast

## Technologies I Used
* Python
* NumPy
* Matplotlib
* TensorFlow
* Keras

## Model Settings
* Hidden layer: 3 neurons
* Activation: Sigmoid
* Output layer: 1 neuron
* Loss: Binary Cross-Entropy
* Optimizer: Adam
* Learning rate: 0.01
* Epochs: 500

## Project Workflow

Data
 ↓
Visualization
 ↓
Normalization
 ↓
Build Neural Network
 ↓
Train Model
 ↓
Check Weights & Biases
 ↓
Test with New Data
 ↓
Make Predictions

## Note
This project uses a small dataset created for learning purposes. The goal of this project was to understand how a simple neural network works rather than to build a real-world coffee roasting system.

This was one of my learning projects as I continue building my foundation in **Machine Learning and AI**.
