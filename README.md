# Neural Network Implementation in Go

This repository contains a Go (Golang) implementation of a neural network, inspired by the concepts in the book *Neural Networks from Scratch in Python* by Sentdex. The goal is to translate these ideas into Go, focusing on a simple yet functional neural network designed to identify patterns in the CAN dataset.

## Overview

This project implements a neural network capable of classifying data from the CAN dataset. It covers essential neural network topics such as:

- Building a feedforward neural network from scratch
- Implementing backpropagation and gradient descent for learning
- Training and testing the network using the CAN dataset
- Handling activation functions like ReLU and Softmax
- Evaluating model performance with metrics like accuracy, loss

## Features

- **Simple Feedforward Neural Network**: A basic neural network architecture, configurable with various layers and neurons.
- **Activation Functions**: Implementation of essential activation functions like ReLU, Sigmoid, and Softmax.
- **Backpropagation and Gradient Descent**: Algorithms to optimize the network’s weights based on loss.
- **Training on CAN Dataset**: Code to train the neural network on the IDS dataset for intrusion detection tasks.
- **Evaluation and Metrics**: Tools to assess the model's performance, focusing on accuracy and loss.

## Getting Started

### Prerequisites

- **Go**: Make sure Go is installed on your system. Download it from the [official website](https://golang.org/dl/).

### Installation

Clone this repository:

```bash
git clone https://github.com/saent-x/IDS-NN.git
cd IDS-NN
```

Install dependencies:

```bash
go mod tidy
```

### Running the Project

To run the neural network with the CAN dataset:

```bash
go run main.go
```