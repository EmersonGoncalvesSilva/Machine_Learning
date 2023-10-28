# Neural Network for Multiclass Classification with Softmax - Implementation from Scratch

## Overview

This Python module provides a comprehensive implementation of a neural network for multiclass classification using the softmax activation function. This neural network is designed to take a set of input features and classify them into one of multiple classes. The softmax function is used to convert the raw output scores of the network into class probabilities.

## Table of Contents

- [Project Name](#project-name)
  - [Table of Contents](#table-of-contents)
  - [Getting Started](#getting-started)
  - [Features](#features)
  - [Concepts](#concepts)
    - [Perceptron](#perceptron)
    - [Activation Function](#Activation-Function)
    - [Layers](Layers#)
    - [Forward Propagation](#Forward-Propagation)
    - [Cost function](#Cost-function)
    - [Minimization of the Loss Optimization Gradient Descent](#Minimization-of-the-Loss-Optimization-Gradient-Descent)
    - [Backward Propagation](#Backward-Propagation)
    - [Derivatives](#Derivatives)
  - [Acknowledgments](#acknowledgments)

## Getting Started

How to create a neural network for multiclass classification:
```Python

>>import neural_network_softmax

# Create a hidden layer:
>> layer_01 = neural_network_softmax.NN_Layer(layer='layer_01', units= 5, activation_func='relu')
# Create final layer with softmax activation function
>> layer_02 = neural_network_softmax.NN_Layer(layer='layer_04', units= 3, activation_func='softmax')

# Create model and connect the layers:
>> nn_class_model = neural_network_softmax.NN([layer_01, layer_02])

# Train model:
>> nn_class_model.train(features=X, labels=Y, alpha= .001)


```


## Features

- Supports any number of input features and classes.
- Uses the softmax activation function for multiclass classification.
- Allows customization of hyperparameters like learning rate, batch size, and the number of hidden layers and neurons.
- Dense hidden layers.
- Activation Functions Options: 
    - `linear`: linear regression funtion
    - `sigmoid`: Logistic regression function
    - `ReLU`: Rectified linear unit
    - `Softmax`: Softmax activation function for multi-class classification
- Optimization Algorithm: `Gradient Descent`
- Cost Function: `Log loss with Cross entropy`.
- About the parementers initialization(`weights` and `bias`):
    - Weights are initialized with value equal to 1. Bias are initalized with values equal to 0. 


## Concepts

How does a `Artificial Neural Network with softmax neural network work`?

### Perceptron
Perceptrons are the building block of the neural network. <br>
Given an input $x_i$ the perceptron will compute a new value $z_i$, as follows:
```math
z_i = w_i x_i + b
```
Where $w_i$ is the weight paremeter, and $b$ is the bias. 
### Activation Function
The equation above is a simple linear transformation, and in order to incorporate allow the model to map non-linearity, activations functions are used in neural network.
- General activation function:
```math
a_i = f(z_i)
```

- Relu activation function.
```math
ReLU(z_i) = max(0, z_i)
```
- Softmax activation function
```math
\text{Softmax}(z)_i = \frac{e^{z_i}}{\sum_{j=1}^{N} e^{z_j}}
```



### Layers
A stack of perceptrons that receive the same input and use the same activation function constitute a layer in the Neural net. 
A neural net can have many layers.

### Forward Propagation
```math
[X_0] -> [A_1]-> [A_2] -> ... -> [Y_{hat}]
```

Forward propagation is the process of passing input data through the neural network to make predictions.
1. Weighted Sum: Calculate the weighted sum of inputs for each neuron in a layer.

2. Activation Function: Apply an activation function (e.g., ReLU) to the weighted sum to introduce non-linearity.

3. Repeat: Repeat these steps for each layer, passing the output of one layer as input to the next.

4. Output Layer: In the output layer, apply the Softmax function to obtain class probabilities.

### Cost function
Once the data goes through all the transformations in the layers of the neural net, the values in the output layer are confronted with the correct values the model should give. The difference between last layer output and the desired results is the model error. The error is calculated using the loss function. 
For multiclass classification problems the loss function typically used is called Cross-Entropy Loss. The cross-entropy loss measures the dissimilarity between the predicted probabilities and the true distribution of the labels.

For a single data point with true label $y$ and predicted probabilities $\hat{y}$, the cross-entropy loss (also known as the log loss) is calculated as follows:

$$
L(y, \hat{y}) = -\sum_{i=1}^{N} y_i \log(\hat{y}_i)
$$

### Minimization of the Loss Optimization Gradient Descent

Ideally the model should have a loss ~ 0. Drive the loss value of the equation above to zero is a task of optimization.

The output of the last layer depends of the paremeters $w$ and $b$ used in each perceptron. Thus minimize the Loss value means find the values of  $w$ and $b$  in each perceptron that leads the the minumum loss.

This tak is done using calculus (derivatives). The objective is to derive the loss in respect to each  $w$ and $b$. The derivate of the loss in respect to the paremeters contains the direction in which they should be updated in order to decrease the loss.
The algorithm to finding the derivatives and progressivelly update the paremeters in order to decrease the loss is called Gradient Descent.   

### Backward Propagation
In order to calculate the derivates of the loss in respect to each $w$ and $b$ the algorithm needs to propagates the calculation from the last layers, where the loss in calcualted, untill the first layer. All of this calcualtions are done used the chain rule of derivation from calculus. 
This whole process is called back propagation.

1. Compute Loss: Calculate the loss (e.g., cross-entropy loss) between predicted and actual values.

2. Backpropagate Error: Compute the gradients of the loss with respect to the model's weights using the chain rule.

3. Update Weights: Update the weights in the network using optimization algorithms like gradient descent.



### Derivatives

#### Main Derivatives to perform back Propagation Using Softmax Activation Function
#### Derivative of Cross-Entropy Loss with Respect to Softmax Input ($(z_i$))

In multiclass classification, we often use the cross-entropy loss to measure the dissimilarity between predicted class probabilities and the true class labels. Given the softmax activation function $(\sigma(z_i)$) for class $(i$) and the true class probability $(y_i$), the cross-entropy loss is defined as:

$$ \mathcal{L}(y_i, \sigma(z_i)) = -y_i \log(\sigma(z_i)) $$

Now, let's calculate the derivative of the loss $(\mathcal{L}$) with respect to the input $(z_i$) of the softmax activation function:

Using the chain rule, we can express this derivative as:

$$ \frac{d\mathcal{L}(y_i, \sigma(z_i))}{dz_i} = \frac{d\mathcal{L}(y_i, \sigma(z_i))}{d\sigma(z_i)} \cdot \frac{d\sigma(z_i)}{dz_i} $$

First, we find the derivative of the loss with respect to \(\sigma(z_i)\):

$$ \frac{d\mathcal{L}(y_i, \sigma(z_i))}{d\sigma(z_i)} = -\frac{y_i}{\sigma(z_i)} $$

Next, we've already calculated the derivative of the softmax activation with respect to its input $(z_i$) as:

$$ \frac{d\sigma(z_i)}{dz_i} = \sigma(z_i) \cdot (1 - \sigma(z_i)) $$

Now, we can combine these results:

$$ \frac{d\mathcal{L}(y_i, \sigma(z_i))}{dz_i} = -\frac{y_i}{\sigma(z_i)} \cdot \sigma(z_i) \cdot (1 - \sigma(z_i)) $$

Simplifying further:

$$ \frac{d\mathcal{L}(y_i, \sigma(z_i))}{dz_i} = -y_i \cdot (1 - \sigma(z_i)) $$

So, the derivative of the cross-entropy loss with respect to the input $(z_i$) of the softmax activation function is:

$$ \frac{d\mathcal{L}(y_i, \sigma(z_i))}{dz_i} = \sigma(z_i) - y_i $$



## Acknowledgments

It was very fun write this project and  exercise my skills in Machine Learning.

