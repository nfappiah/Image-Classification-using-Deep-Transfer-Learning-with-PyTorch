# Image Classification using Deep Transfer Learning

In this project, I implemented a deep learning image classifier capable of recognizing different species of flowers using PyTorch and Jupyter notebook. I built and trained the deep neural network via transfer learning and then used the trained network to predict the class for an input image.

## Dataset

The dataset, consisting of 102 flower categories, can be found at: 
[https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html)

## Methods

The methods employed in this project involve:
* Loading and preprocessing the images
* Training the deep learning image classifier on the images
* Using the trained classifier to predict new images

## Deep Learning Architecture and Transfer Learning

Alexnet, a pre-trained network loaded from torchvision.models, was used as the deep learning architecture. I defined a new, untrained feed-forward neural network as a classifier, using ReLU activations and dropout. I trained the classifier layers using backpropagation using the pre-trained network and tracked the loss and accuracy on the validation set to determine the best hyperparameters. 

## Materials and Packages

* Jupyter Notebook
* Python 3.8.8
* matplotlib
* torch 2.0.1
* torchvision
* collections
* PIL
* numpy
* random
* json
* os

## GPU

The notebook is set up to use GPU if it is available.

## Instructions 

* To run the jupyter notebook, users must have the required dataset in the working directory (same directory as the jupyter notebook). Be sure to update the PATH in the first code cell
* The flower directory must contain 3 sub directories: test, train, and valid. Each of the test, train and valid sub directories must contain further sub directories corresponding to the image categories. Each image category sub directory must contain images only belonging to that category. It is vital that the dataset is set up in this way for notebook to run.
* Users must also have the 'cat_to_name.json' file in the working directory. The 'cat_to_name.json' file is a dictionary that maps category label (number or digit) to category (flower) name. If you would like to run this notebook, you must generate this file. The categories can be found in the link to dataset provided above.
* The entire notebook can be run at once by pressing Ctrl + F9. However, it is recommended to run the notebook one cell at a time to observe the output. A code cell can be run by selecting it and pressing Ctrl+Shift+Enter or Shift+Enter

## Acknowledgements

I am grateful to AWS for awarding me the AWS AI & Machine Learning Scholarhip to pursue the nanodegree in "AI Programming with Python" through Udacity. This Jupyter notebook represents one of the final projects I undertook as part of the requirements to complete the nanodegree.