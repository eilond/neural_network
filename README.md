# Project Name

## Overview
## Overview

This project involves the development and implementation of a neural network for the purpose of classification tasks, focusing on aspects such as backpropagation, optimization, and the influence of hyperparameters. The project serves as a hands-on experience in scientific programming and the development of complex optimization frameworks.

The core objective is to define, develop, and train a neural network capable of classifying small vector data samples. This involves working with labeled training data samples $(x_i, y_i)$ and a separate set of test data $(x_{ti}, y_{ti})$, aiming to optimize the network to accurately predict labels for given data samples. 

Key aspects of the project include:
- **Network Architecture**: The network's architecture is structured to be either a standard Neural Network with the formulation $f(θ^{(l)}, x^{(l)}_i) = σ(W^{(l)}x^{(l)}_i + b^{(l)})$, or a simple ResNet. The architecture emphasizes the dynamic learning of weights through various layers.
- **Softmax Objective Function**: Implementation of the softmax objective function, as defined in the lectures, plays a crucial role in predicting the label with the highest probability for each data sample.
- **Layer Weights Optimization**: The weights of each layer $θ^{(l)}$ are key parameters to be learned during the training process. This includes learning softmax weights and bias vectors for the last layer and appropriate weight matrices and biases for standard and residual layers.
- **Expanding Network Width**: The project explores the common practice of expanding the width of the network, i.e., the size of the vector $x^{(l)}_i$ as the layers progress, especially for the residual layers.



## Features
#TODO

## Installation
#TODO

```bash
#TODO
